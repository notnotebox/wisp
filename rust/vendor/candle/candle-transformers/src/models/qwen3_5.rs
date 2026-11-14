use crate::models::with_tracing::{linear_b, linear_no_bias, Linear};
use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{conv1d_no_bias, Activation, Conv1d, Conv1dConfig, VarBuilder};
use std::cell::RefCell;
use std::sync::Arc;
use std::time::Instant;

thread_local! {
    static PROFILE: RefCell<Vec<(&'static str, f64, u32)>> = const { RefCell::new(Vec::new()) };
}

fn profiling() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("WISP_CANDLE_PROFILE").is_some())
}

/// With WISP_CANDLE_PROFILE=1, waits for the GPU before and after `f` so the time is the section's own.
fn timed<T>(name: &'static str, dev: &Device, f: impl FnOnce() -> Result<T>) -> Result<T> {
    if !profiling() {
        return f();
    }
    dev.synchronize()?;
    let start = Instant::now();
    let out = f()?;
    dev.synchronize()?;
    let ms = start.elapsed().as_secs_f64() * 1e3;
    PROFILE.with(|p| {
        let mut p = p.borrow_mut();
        match p.iter_mut().find(|e| e.0 == name) {
            Some(e) => {
                e.1 += ms;
                e.2 += 1;
            }
            None => p.push((name, ms, 1)),
        }
    });
    Ok(out)
}

/// Per-section GPU time since the last call (empty unless WISP_CANDLE_PROFILE is set).
pub fn take_profile() -> String {
    PROFILE.with(|p| {
        let mut p = p.borrow_mut();
        let line = p
            .iter()
            .map(|(name, ms, calls)| format!("{name}={ms:.0}ms/{calls}"))
            .collect::<Vec<_>>()
            .join(" ");
        p.clear();
        line
    })
}

// Candle's Windows CUDA backend does not currently implement the generic
// sigmoid op. Express it through primitive CUDA kernels instead.
fn cuda_safe_sigmoid(xs: &Tensor) -> Result<Tensor> {
    (xs.neg()?.exp()? + 1.0)?.recip()
}

// `Tensor::silu` is a core unary op with its own CUDA kernel, so it is one launch instead of
// the five (neg, exp, add, recip, mul) of composing it from `cuda_safe_sigmoid`.
fn cuda_safe_silu(xs: &Tensor) -> Result<Tensor> {
    // WISP_CANDLE_COMPOSED_SILU=1 restores the composed version for parity and speed checks.
    static COMPOSED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *COMPOSED.get_or_init(|| std::env::var_os("WISP_CANDLE_COMPOSED_SILU").is_some()) {
        return xs.broadcast_mul(&cuda_safe_sigmoid(xs)?);
    }
    xs.silu()
}

// Equivalent to candle_nn::rotary_emb::rope, expressed with ordinary tensor
// operations because the generic half-split rotary custom op has no Windows
// CUDA implementation yet.
fn cuda_safe_rope(xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let last_dim = xs.dim(D::Minus1)?;
    let first = xs.narrow(D::Minus1, 0, last_dim / 2)?;
    let second = xs.narrow(D::Minus1, last_dim / 2, last_dim - last_dim / 2)?;
    let rotated = Tensor::cat(&[&second.neg()?, &first], D::Minus1)?;
    let cos = Tensor::cat(&[cos, cos], D::Minus1)?.unsqueeze(0)?.unsqueeze(0)?;
    let sin = Tensor::cat(&[sin, sin], D::Minus1)?.unsqueeze(0)?.unsqueeze(0)?;
    (xs.broadcast_mul(&cos)? + rotated.broadcast_mul(&sin)?)?.contiguous()
}

/// Depthwise (one filter per channel) convolution over an already padded input.
///
/// `padded`: (batch, channels, len + kernel - 1), `weight`: (channels, 1, kernel).
/// Returns (batch, channels, len). Candle's grouped `conv1d` runs one convolution per group,
/// which is one kernel launch per channel here; this is `kernel` broadcast multiply-adds instead.
pub fn depthwise_conv1d(padded: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let (_batch, channels, padded_len) = padded.dims3()?;
    let kernel = weight.dim(2)?;
    let len = padded_len + 1 - kernel;
    let dtype = padded.dtype();
    let padded = padded.to_dtype(DType::F32)?;
    let weight = weight.to_dtype(DType::F32)?.reshape((1, channels, kernel))?;
    let mut out = padded
        .narrow(2, 0, len)?
        .broadcast_mul(&weight.narrow(2, 0, 1)?)?;
    for tap in 1..kernel {
        out = (out + padded.narrow(2, tap, len)?.broadcast_mul(&weight.narrow(2, tap, 1)?)?)?;
    }
    out.to_dtype(dtype)
}

fn cuda_safe_softmax_last_dim(xs: &Tensor) -> Result<Tensor> {
    let shifted = xs.broadcast_sub(&xs.max_keepdim(D::Minus1)?)?;
    let numerators = shifted.exp()?;
    numerators.broadcast_div(&numerators.sum_keepdim(D::Minus1)?)
}

/// A linear projection inside a Qwen 3.5 layer.
///
/// Implement this trait to load projections whose weights are stored in a
/// format other than Candle's dense tensors, while retaining the model's
/// attention, linear-attention, norms, cache handling, and residual wiring.
pub trait Projection: Send + Sync {
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}

impl<T: Module + Send + Sync> Projection for T {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Module::forward(self, xs)
    }
}

#[derive(Clone)]
enum ProjectionImpl {
    Dense(Linear),
    Custom(Arc<dyn Projection>),
}

impl std::fmt::Debug for ProjectionImpl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dense(linear) => linear.fmt(f),
            Self::Custom(_) => f.write_str("dyn Projection"),
        }
    }
}

impl ProjectionImpl {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(linear) => Module::forward(linear, xs),
            Self::Custom(projection) => projection.forward(xs),
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct Config {
    pub text_config: TextConfig,
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: Option<usize>,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_parameters: RopeParameters,
    pub attention_bias: bool,
    pub hidden_act: Activation,

    // Sparse-MoE checkpoints use these instead of the dense intermediate size.
    #[serde(default)]
    pub num_experts: usize,
    #[serde(default)]
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub moe_intermediate_size: usize,
    #[serde(default)]
    pub shared_expert_intermediate_size: usize,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f64,

    // Qwen 3.5 specific
    pub layer_types: Vec<String>,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
}

fn default_partial_rotary_factor() -> f64 {
    1.
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f64,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.text_config
            .head_dim
            .unwrap_or(self.text_config.hidden_size / self.text_config.num_attention_heads)
    }

    pub fn rope_theta(&self) -> f64 {
        self.text_config.rope_parameters.rope_theta
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3_5RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl Qwen3_5RmsNorm {
    pub fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let hidden_states = x.to_dtype(internal_dtype)?;
        let variance = hidden_states.sqr()?.mean_keepdim(D::Minus1)?;
        let hidden_states =
            hidden_states.broadcast_mul(&(&variance + self.eps)?.sqrt()?.recip()?)?;
        let weight = (&self.weight.to_dtype(internal_dtype)? + 1.0)?;
        hidden_states.broadcast_mul(&weight)?.to_dtype(x_dtype)
    }
}

fn l2_norm(x: &Tensor) -> Result<Tensor> {
    let inv_norm = (x.sqr()?.sum_keepdim(D::Minus1)? + 1e-6)?.sqrt()?.recip()?;
    x.broadcast_mul(&inv_norm)
}

fn repeat_interleave(x: &Tensor, n: usize, dim: usize) -> Result<Tensor> {
    if n == 1 {
        return Ok(x.clone());
    }
    let mut dims = x.dims().to_vec();
    dims.insert(dim + 1, n);
    let expanded = x.unsqueeze(dim + 1)?.broadcast_as(dims.as_slice())?;
    let mut final_dims = x.dims().to_vec();
    final_dims[dim] *= n;
    expanded.reshape(final_dims.as_slice())
}

#[derive(Debug, Clone)]
pub struct Qwen3_5GatedDeltaNet {
    num_v_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    conv_kernel_size: usize,
    conv_dim: usize,

    conv1d: Conv1d,
    dt_bias_f32: Tensor,
    neg_a_f32: Tensor,

    in_proj_qkv: ProjectionImpl,
    in_proj_z: ProjectionImpl,
    in_proj_b: ProjectionImpl,
    in_proj_a: ProjectionImpl,
    out_proj: ProjectionImpl,

    norm_weight: Tensor,
    norm_eps: f64,

    conv_state: Option<Tensor>,
    recurrent_state: Option<Tensor>,
}

impl Qwen3_5GatedDeltaNet {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = cfg.text_config.hidden_size;
        let num_v_heads = cfg.text_config.linear_num_value_heads;
        let num_k_heads = cfg.text_config.linear_num_key_heads;
        let head_k_dim = cfg.text_config.linear_key_head_dim;
        let head_v_dim = cfg.text_config.linear_value_head_dim;
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;

        let conv_dim = key_dim * 2 + value_dim;
        let conv_kernel_size = cfg.text_config.linear_conv_kernel_dim;

        let conv1d_cfg = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: conv_dim,
            ..Default::default()
        };
        // wait, we manually pad, so padding is 0 for the conv1d module itself
        let conv1d = conv1d_no_bias(
            conv_dim,
            conv_dim,
            conv_kernel_size,
            conv1d_cfg,
            vb.pp("conv1d"),
        )?;

        let dt_bias = vb.get(num_v_heads, "dt_bias")?;
        let a_log = vb.get(num_v_heads, "A_log")?;

        let dt_bias_f32 = dt_bias.to_dtype(DType::F32)?;
        let neg_a_f32 = (&a_log.to_dtype(DType::F32)?.exp()? * -1.0)?;

        let in_proj_qkv = linear_no_bias(hidden_size, conv_dim, vb.pp("in_proj_qkv"))?;
        let in_proj_z = linear_no_bias(hidden_size, value_dim, vb.pp("in_proj_z"))?;
        let in_proj_b = linear_no_bias(hidden_size, num_v_heads, vb.pp("in_proj_b"))?;
        let in_proj_a = linear_no_bias(hidden_size, num_v_heads, vb.pp("in_proj_a"))?;
        let out_proj = linear_no_bias(value_dim, hidden_size, vb.pp("out_proj"))?;

        let norm_weight = vb.get(head_v_dim, "norm.weight")?;

        Ok(Self {
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            conv_kernel_size,
            conv_dim,
            conv1d,
            dt_bias_f32,
            neg_a_f32,
            in_proj_qkv: ProjectionImpl::Dense(in_proj_qkv),
            in_proj_z: ProjectionImpl::Dense(in_proj_z),
            in_proj_b: ProjectionImpl::Dense(in_proj_b),
            in_proj_a: ProjectionImpl::Dense(in_proj_a),
            out_proj: ProjectionImpl::Dense(out_proj),
            norm_weight,
            norm_eps: cfg.text_config.rms_norm_eps,
            conv_state: None,
            recurrent_state: None,
        })
    }

    fn new_with_projections<F>(cfg: &Config, vb: VarBuilder, projections: &F) -> Result<Self>
    where
        F: for<'a> Fn(&str, VarBuilder<'a>, usize, usize, bool) -> Result<Box<dyn Projection>>,
    {
        let hidden_size = cfg.text_config.hidden_size;
        let num_v_heads = cfg.text_config.linear_num_value_heads;
        let num_k_heads = cfg.text_config.linear_num_key_heads;
        let head_k_dim = cfg.text_config.linear_key_head_dim;
        let head_v_dim = cfg.text_config.linear_value_head_dim;
        let key_dim = head_k_dim * num_k_heads;
        let value_dim = head_v_dim * num_v_heads;
        let conv_dim = key_dim * 2 + value_dim;
        let conv_kernel_size = cfg.text_config.linear_conv_kernel_dim;
        let conv1d = conv1d_no_bias(
            conv_dim,
            conv_dim,
            conv_kernel_size,
            Conv1dConfig {
                padding: 0,
                stride: 1,
                dilation: 1,
                groups: conv_dim,
                ..Default::default()
            },
            vb.pp("conv1d"),
        )?;
        let dt_bias_f32 = vb.get(num_v_heads, "dt_bias")?.to_dtype(DType::F32)?;
        let neg_a_f32 = (&vb.get(num_v_heads, "A_log")?.to_dtype(DType::F32)?.exp()? * -1.0)?;
        let in_proj_qkv = projections("in_proj_qkv", vb.clone(), hidden_size, conv_dim, false)?;
        let in_proj_z = projections("in_proj_z", vb.clone(), hidden_size, value_dim, false)?;
        let in_proj_b = projections("in_proj_b", vb.clone(), hidden_size, num_v_heads, false)?;
        let in_proj_a = projections("in_proj_a", vb.clone(), hidden_size, num_v_heads, false)?;
        let out_proj = projections("out_proj", vb.clone(), value_dim, hidden_size, false)?;
        let norm_weight = vb.get(head_v_dim, "norm.weight")?;
        Ok(Self {
            num_v_heads,
            num_k_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            conv_kernel_size,
            conv_dim,
            conv1d,
            dt_bias_f32,
            neg_a_f32,
            in_proj_qkv: ProjectionImpl::Custom(Arc::from(in_proj_qkv)),
            in_proj_z: ProjectionImpl::Custom(Arc::from(in_proj_z)),
            in_proj_b: ProjectionImpl::Custom(Arc::from(in_proj_b)),
            in_proj_a: ProjectionImpl::Custom(Arc::from(in_proj_a)),
            out_proj: ProjectionImpl::Custom(Arc::from(out_proj)),
            norm_weight,
            norm_eps: cfg.text_config.rms_norm_eps,
            conv_state: None,
            recurrent_state: None,
        })
    }

    /// WISP_CANDLE_GROUPED_CONV=1 keeps Candle's per-group conv1d for parity checks.
    fn depthwise_conv(&self, padded: &Tensor) -> Result<Tensor> {
        if std::env::var_os("WISP_CANDLE_GROUPED_CONV").is_some() {
            return padded.conv1d(self.conv1d.weight(), 0, 1, 1, self.conv_dim);
        }
        depthwise_conv1d(padded, self.conv1d.weight())
    }

    fn rms_norm_gated(&self, hidden_states: &Tensor, gate: &Tensor) -> Result<Tensor> {
        let input_dtype = hidden_states.dtype();
        let hidden_states = hidden_states.to_dtype(DType::F32)?;
        let variance = hidden_states.sqr()?.mean_keepdim(D::Minus1)?;
        let hidden_states =
            hidden_states.broadcast_mul(&(&variance + self.norm_eps)?.sqrt()?.recip()?)?;
        let hidden_states = hidden_states.broadcast_mul(&self.norm_weight.to_dtype(DType::F32)?)?;
        let hidden_states = hidden_states.to_dtype(input_dtype)?;
        let gate_silu = cuda_safe_silu(&gate.to_dtype(DType::F32)?)?.to_dtype(input_dtype)?;
        hidden_states.broadcast_mul(&gate_silu)
    }

    fn torch_recurrent_gated_delta_rule(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        g: &Tensor,
        beta: &Tensor,
        initial_state: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let query = l2_norm(query)?;
        let key = l2_norm(key)?;

        // Prompt prefill: the chunked scan needs O(seq_len / 64) sequential steps instead of one
        // step per token. WISP_CANDLE_SEQUENTIAL_SCAN=1 keeps the per-token loop for parity checks.
        if query.dim(1)? > 1 && std::env::var_os("WISP_CANDLE_SEQUENTIAL_SCAN").is_none() {
            let (batch_size, _seq_len, num_heads, k_head_dim) = key.dims4()?;
            let v_head_dim = value.dim(3)?;
            let scale = 1.0 / (k_head_dim as f64).sqrt();
            let q = (query.to_dtype(DType::F32)? * scale)?.contiguous()?;
            let k = key.to_dtype(DType::F32)?.contiguous()?;
            let v = value.to_dtype(DType::F32)?.contiguous()?;
            // g is already the log of the per-step decay (the loop below applies g.exp()).
            let log_g = g.to_dtype(DType::F32)?.contiguous()?;
            let beta = beta.to_dtype(DType::F32)?.contiguous()?;
            let mut state = match initial_state {
                Some(state) => state.to_dtype(DType::F32)?,
                None => Tensor::zeros(
                    (batch_size, num_heads, k_head_dim, v_head_dim),
                    DType::F32,
                    q.device(),
                )?,
            };
            use crate::models::qwen3_5_linear_attn_scan as scan;
            // WISP_CANDLE_NO_FUSED_PREFILL=1 keeps the pure-Candle chunked scan for parity checks.
            let fused = scan::fused_prefill_supported(&q, &v)
                && std::env::var_os("WISP_CANDLE_NO_FUSED_PREFILL").is_none();
            let out = if fused {
                scan::gated_delta_rule_fused_prefill(&q, &k, &v, &log_g, &beta, &mut state)?
            } else {
                scan::gated_delta_rule_chunked(&q, &k, &v, &log_g, &beta, &mut state)?
            }; // (B, S, H, V)
            return Ok((out, state));
        }

        // Keep in initial_dtype (usually BF16) if possible, but core update might need F32 for stability
        let query = query.transpose(1, 2)?; // (batch, heads, seq, dim)
        let key = key.transpose(1, 2)?;
        let value = value.transpose(1, 2)?;
        let beta = beta.transpose(1, 2)?;
        let g = g.transpose(1, 2)?;

        let (batch_size, num_heads, seq_len, k_head_dim) = key.dims4()?;
        let v_head_dim = value.dim(3)?;

        let scale = 1.0 / (query.dim(3)? as f64).sqrt();
        let query = (query * scale)?;

        let mut core_attn_out_vec = Vec::with_capacity(seq_len);

        // We MUST use F32 for the recurrent state to avoid numerical drift
        let mut last_recurrent_state = match initial_state {
            Some(state) => state.to_dtype(DType::F32)?,
            None => Tensor::zeros(
                (batch_size, num_heads, k_head_dim, v_head_dim),
                DType::F32,
                query.device(),
            )?,
        };

        // Move loop-invariant conversions outside and prepare dimensions
        let query_f32 = query.to_dtype(DType::F32)?;
        let key_f32 = key.to_dtype(DType::F32)?;
        let value_f32 = value.to_dtype(DType::F32)?;
        let g_exp = g.to_dtype(DType::F32)?.exp()?.unsqueeze(3)?; // (B, H, S, 1)
        let beta_f32 = beta.to_dtype(DType::F32)?.unsqueeze(3)?; // (B, H, S, 1)

        for i in 0..seq_len {
            let q_t = query_f32.narrow(2, i, 1)?; // (B, H, 1, K)
            let k_t = key_f32.narrow(2, i, 1)?; // (B, H, 1, K)
            let v_t = value_f32.narrow(2, i, 1)?; // (B, H, 1, V)
            let g_t = g_exp.narrow(2, i, 1)?; // (B, H, 1, 1)
            let beta_t = beta_f32.narrow(2, i, 1)?; // (B, H, 1, 1)

            last_recurrent_state = last_recurrent_state.broadcast_mul(&g_t)?;

            // k_t.matmul(&state) is equivalent to broadcast_mul + sum(2)
            let kv_mem = k_t.matmul(&last_recurrent_state)?; // (B, H, 1, V)
            let delta = (&v_t - &kv_mem)?.broadcast_mul(&beta_t)?; // (B, H, 1, V)

            // k_t.T.matmul(&delta) is equivalent to outer product
            let delta_k = k_t.transpose(2, 3)?.matmul(&delta)?; // (B, H, K, 1) x (B, H, 1, V) = (B, H, K, V)
            last_recurrent_state = (&last_recurrent_state + &delta_k)?;

            let out_t = q_t.matmul(&last_recurrent_state)?; // (B, H, 1, V)
            core_attn_out_vec.push(out_t);
        }

        let core_attn_out = Tensor::cat(&core_attn_out_vec, 2)?; // (B, H, S, V)
        let core_attn_out = core_attn_out.transpose(1, 2)?; // (B, S, H, V)
        Ok((core_attn_out, last_recurrent_state))
    }

    pub fn forward(&mut self, hidden_states: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;
        let hidden_states = hidden_states.contiguous()?;
        let initial_dtype = hidden_states.dtype();

        let dev = hidden_states.device().clone();
        let (mixed_qkv, z, b, a) = timed("lin_in_proj", &dev, || {
            let mixed_qkv = self.in_proj_qkv.forward(&hidden_states)?.transpose(1, 2)?;
            let z = self.in_proj_z.forward(&hidden_states)?;
            let z = z.reshape((batch_size, seq_len, (), self.head_v_dim))?;
            let b = self.in_proj_b.forward(&hidden_states)?;
            let a = self.in_proj_a.forward(&hidden_states)?;
            Ok((mixed_qkv, z, b, a))
        })?;

        let use_precomputed_states = self.conv_state.is_some() && seq_len == 1;

        let mixed_qkv = timed("lin_conv", &dev, || -> Result<Tensor> { Ok(if use_precomputed_states {
            let conv_state = self.conv_state.as_mut().unwrap();
            let conv_state_data = Tensor::cat(&[conv_state, &mixed_qkv], 2)?;
            *conv_state = conv_state_data.narrow(2, 1, self.conv_kernel_size - 1)?;
            let out = self.depthwise_conv(&conv_state_data)?;
            cuda_safe_silu(&out)?
        } else {
            let pad = self.conv_kernel_size - 1;
            let padding = Tensor::zeros(
                (batch_size, self.conv_dim, pad),
                mixed_qkv.dtype(),
                mixed_qkv.device(),
            )?;
            let padded_qkv = Tensor::cat(&[&padding, &mixed_qkv], 2)?;
            self.conv_state = Some(padded_qkv.narrow(2, seq_len, pad)?);
            let out = self.depthwise_conv(&padded_qkv)?;
            cuda_safe_silu(&out)?
        }) })?;

        let mixed_qkv = mixed_qkv.transpose(1, 2)?; // (batch, seq, conv_dim)

        let q = mixed_qkv.narrow(D::Minus1, 0, self.key_dim)?;
        let k = mixed_qkv.narrow(D::Minus1, self.key_dim, self.key_dim)?;
        let v = mixed_qkv.narrow(D::Minus1, self.key_dim * 2, self.value_dim)?;

        let q = q.reshape((batch_size, seq_len, (), self.head_k_dim))?;
        let k = k.reshape((batch_size, seq_len, (), self.head_k_dim))?;
        let v = v.reshape((batch_size, seq_len, (), self.head_v_dim))?;

        let beta = cuda_safe_sigmoid(&b)?;
        let g = {
            let a_f32 = a.to_dtype(DType::F32)?;
            let a_plus_dt = a_f32.broadcast_add(&self.dt_bias_f32)?;
            let softplus = (a_plus_dt.exp()? + 1.0)?.log()?;
            self.neg_a_f32.broadcast_mul(&softplus)?
        };

        let repeat_n = self.num_v_heads / self.num_k_heads;
        let q = repeat_interleave(&q, repeat_n, 2)?;
        let k = repeat_interleave(&k, repeat_n, 2)?;

        let initial_state = if use_precomputed_states {
            self.recurrent_state.as_ref()
        } else {
            None
        };

        let (core_attn_out, new_state) = timed("lin_scan", &dev, || {
            self.torch_recurrent_gated_delta_rule(&q, &k, &v, &g, &beta, initial_state)
        })?;
        self.recurrent_state = Some(new_state);

        let core_attn_out = core_attn_out.to_dtype(initial_dtype)?;
        let core_attn_out = core_attn_out.reshape(((), self.head_v_dim))?;
        let z_flat = z.reshape(((), self.head_v_dim))?;
        let core_attn_out = timed("lin_gate_norm", &dev, || self.rms_norm_gated(&core_attn_out, &z_flat))?;
        let core_attn_out = core_attn_out.reshape((batch_size, seq_len, ()))?;

        timed("lin_out_proj", &dev, || self.out_proj.forward(&core_attn_out))
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3_5TextRotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    rotary_dim: usize,
}

impl Qwen3_5TextRotaryEmbedding {
    pub fn new(dtype: DType, cfg: &Config, dev: &Device) -> Result<Self> {
        let dim = (cfg.head_dim() as f64 * cfg.text_config.partial_rotary_factor) as usize;
        if dim == 0 || dim > cfg.head_dim() || !dim.is_multiple_of(2) {
            candle::bail!(
                "invalid Qwen 3.5 rotary dimension {dim} for head dimension {}",
                cfg.head_dim()
            )
        }
        let max_seq_len = cfg.text_config.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta().powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
            rotary_dim: dim,
        })
    }

    pub fn apply(&self, q: &Tensor, k: &Tensor, seqlen_offset: usize) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let apply = |x: &Tensor| -> Result<Tensor> {
            let rotated = cuda_safe_rope(
                &x.narrow(D::Minus1, 0, self.rotary_dim)?.contiguous()?,
                &cos,
                &sin,
            )?;
            if self.rotary_dim == x.dim(D::Minus1)? {
                Ok(rotated)
            } else {
                Tensor::cat(
                    &[
                        &rotated,
                        &x.narrow(
                            D::Minus1,
                            self.rotary_dim,
                            x.dim(D::Minus1)? - self.rotary_dim,
                        )?,
                    ],
                    D::Minus1,
                )
            }
        };
        let q_embed = apply(q)?;
        let k_embed = apply(k)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3_5Attention {
    q_proj: ProjectionImpl,
    k_proj: ProjectionImpl,
    v_proj: ProjectionImpl,
    o_proj: ProjectionImpl,
    q_norm: Qwen3_5RmsNorm,
    k_norm: Qwen3_5RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<Qwen3_5TextRotaryEmbedding>,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Qwen3_5Attention {
    pub fn new(
        cfg: &Config,
        rotary_emb: Arc<Qwen3_5TextRotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = cfg.head_dim();
        let num_heads = cfg.text_config.num_attention_heads;
        let num_kv_heads = cfg.text_config.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;

        let q_proj = linear_b(
            cfg.text_config.hidden_size,
            num_heads * head_dim * 2,
            cfg.text_config.attention_bias,
            vb.pp("q_proj"),
        )?;
        let k_proj = linear_b(
            cfg.text_config.hidden_size,
            num_kv_heads * head_dim,
            cfg.text_config.attention_bias,
            vb.pp("k_proj"),
        )?;
        let v_proj = linear_b(
            cfg.text_config.hidden_size,
            num_kv_heads * head_dim,
            cfg.text_config.attention_bias,
            vb.pp("v_proj"),
        )?;
        let o_proj = linear_b(
            num_heads * head_dim,
            cfg.text_config.hidden_size,
            cfg.text_config.attention_bias,
            vb.pp("o_proj"),
        )?;

        let q_norm = Qwen3_5RmsNorm::new(head_dim, cfg.text_config.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = Qwen3_5RmsNorm::new(head_dim, cfg.text_config.rms_norm_eps, vb.pp("k_norm"))?;

        Ok(Self {
            q_proj: ProjectionImpl::Dense(q_proj),
            k_proj: ProjectionImpl::Dense(k_proj),
            v_proj: ProjectionImpl::Dense(v_proj),
            o_proj: ProjectionImpl::Dense(o_proj),
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            rotary_emb,
            kv_cache: None,
        })
    }

    fn new_with_projections<F>(
        cfg: &Config,
        rotary_emb: Arc<Qwen3_5TextRotaryEmbedding>,
        vb: VarBuilder,
        projections: &F,
    ) -> Result<Self>
    where
        F: for<'a> Fn(&str, VarBuilder<'a>, usize, usize, bool) -> Result<Box<dyn Projection>>,
    {
        let head_dim = cfg.head_dim();
        let num_heads = cfg.text_config.num_attention_heads;
        let num_kv_heads = cfg.text_config.num_key_value_heads;
        let q_proj = projections(
            "q_proj",
            vb.clone(),
            cfg.text_config.hidden_size,
            num_heads * head_dim * 2,
            cfg.text_config.attention_bias,
        )?;
        let k_proj = projections(
            "k_proj",
            vb.clone(),
            cfg.text_config.hidden_size,
            num_kv_heads * head_dim,
            cfg.text_config.attention_bias,
        )?;
        let v_proj = projections(
            "v_proj",
            vb.clone(),
            cfg.text_config.hidden_size,
            num_kv_heads * head_dim,
            cfg.text_config.attention_bias,
        )?;
        let o_proj = projections(
            "o_proj",
            vb.clone(),
            num_heads * head_dim,
            cfg.text_config.hidden_size,
            cfg.text_config.attention_bias,
        )?;
        let q_norm = Qwen3_5RmsNorm::new(head_dim, cfg.text_config.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = Qwen3_5RmsNorm::new(head_dim, cfg.text_config.rms_norm_eps, vb.pp("k_norm"))?;
        Ok(Self {
            q_proj: ProjectionImpl::Custom(Arc::from(q_proj)),
            k_proj: ProjectionImpl::Custom(Arc::from(k_proj)),
            v_proj: ProjectionImpl::Custom(Arc::from(v_proj)),
            o_proj: ProjectionImpl::Custom(Arc::from(o_proj)),
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            rotary_emb,
            kv_cache: None,
        })
    }

    pub fn forward(
        &mut self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = hidden_states.dims3()?;

        let q_proj_out = self.q_proj.forward(hidden_states)?;
        let q_proj_out = q_proj_out.reshape((b_sz, q_len, self.num_heads, self.head_dim * 2))?;

        let query_states = q_proj_out.narrow(D::Minus1, 0, self.head_dim)?;
        let gate = q_proj_out.narrow(D::Minus1, self.head_dim, self.head_dim)?;

        let query_states = self.q_norm.forward(&query_states)?.transpose(1, 2)?;

        let key_states = self.k_proj.forward(hidden_states)?;
        let key_states = key_states.reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?;
        let key_states = self.k_norm.forward(&key_states)?.transpose(1, 2)?;

        let value_states = self.v_proj.forward(hidden_states)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (query_states, key_states) =
            self.rotary_emb
                .apply(&query_states, &key_states, seqlen_offset)?;

        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[prev_k, &key_states], 2)?;
                let v = Tensor::cat(&[prev_v, &value_states], 2)?;
                (k, v)
            }
        };
        self.kv_cache = Some((key_states.clone(), value_states.clone()));

        let key_states = crate::utils::repeat_kv(key_states, self.num_kv_groups)?.contiguous()?;
        let value_states =
            crate::utils::repeat_kv(value_states, self.num_kv_groups)?.contiguous()?;

        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        // Candle CUDA matmul requires materialized inputs; the rotary path and
        // transposition otherwise leave strided views on Windows.
        let query_states = query_states.contiguous()?;
        let key_states_t = key_states.transpose(2, 3)?.contiguous()?;
        let attn_weights = (query_states.matmul(&key_states_t)? * scale)?;

        let attn_weights = match attention_mask {
            None => attn_weights,
            Some(mask) => attn_weights.broadcast_add(mask)?,
        };
        let attn_weights = cuda_safe_softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&value_states)?;

        let attn_output =
            attn_output
                .transpose(1, 2)?
                .reshape((b_sz, q_len, self.num_heads, self.head_dim))?;

        let attn_output = attn_output.broadcast_mul(
            &cuda_safe_sigmoid(&gate.to_dtype(DType::F32)?)?.to_dtype(attn_output.dtype())?,
        )?;

        let attn_output = attn_output.reshape((b_sz, q_len, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&attn_output)
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3_5MLP {
    gate_proj: ProjectionImpl,
    up_proj: ProjectionImpl,
    down_proj: ProjectionImpl,
    act_fn: Activation,
}

impl Qwen3_5MLP {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Self::new_sized(cfg, cfg.text_config.intermediate_size, vb)
    }

    fn new_sized(cfg: &Config, intermediate_sz: usize, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.text_config.hidden_size;
        let gate_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("gate_proj"))?;
        let up_proj = linear_no_bias(hidden_sz, intermediate_sz, vb.pp("up_proj"))?;
        let down_proj = linear_no_bias(intermediate_sz, hidden_sz, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj: ProjectionImpl::Dense(gate_proj),
            up_proj: ProjectionImpl::Dense(up_proj),
            down_proj: ProjectionImpl::Dense(down_proj),
            act_fn: cfg.text_config.hidden_act,
        })
    }

    fn new_with_projections<F>(cfg: &Config, vb: VarBuilder, projections: &F) -> Result<Self>
    where
        F: for<'a> Fn(&str, VarBuilder<'a>, usize, usize, bool) -> Result<Box<dyn Projection>>,
    {
        Self::new_with_projections_sized(
            cfg,
            cfg.text_config.intermediate_size,
            vb,
            "",
            projections,
        )
    }

    fn new_with_projections_sized<F>(
        cfg: &Config,
        intermediate_sz: usize,
        vb: VarBuilder,
        prefix: &str,
        projections: &F,
    ) -> Result<Self>
    where
        F: for<'a> Fn(&str, VarBuilder<'a>, usize, usize, bool) -> Result<Box<dyn Projection>>,
    {
        let hidden_sz = cfg.text_config.hidden_size;
        let name = |projection: &str| {
            if prefix.is_empty() {
                projection.to_owned()
            } else {
                format!("{prefix}.{projection}")
            }
        };
        let gate_proj = projections(
            &name("gate_proj"),
            vb.clone(),
            hidden_sz,
            intermediate_sz,
            false,
        )?;
        let up_proj = projections(
            &name("up_proj"),
            vb.clone(),
            hidden_sz,
            intermediate_sz,
            false,
        )?;
        let down_proj = projections(&name("down_proj"), vb, intermediate_sz, hidden_sz, false)?;
        Ok(Self {
            gate_proj: ProjectionImpl::Custom(Arc::from(gate_proj)),
            up_proj: ProjectionImpl::Custom(Arc::from(up_proj)),
            down_proj: ProjectionImpl::Custom(Arc::from(down_proj)),
            act_fn: cfg.text_config.hidden_act,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = match self.act_fn {
            Activation::Silu => cuda_safe_silu(&self.gate_proj.forward(xs)?)?,
            _ => self.gate_proj.forward(xs)?.apply(&self.act_fn)?,
        };
        let rhs = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(lhs * rhs)?)
    }
}

#[derive(Debug, Clone)]
struct Qwen3_5SparseMoeBlock {
    gate: ProjectionImpl,
    experts: Vec<Qwen3_5MLP>,
    shared_expert: Qwen3_5MLP,
    shared_expert_gate: ProjectionImpl,
    num_experts_per_tok: usize,
}

impl Qwen3_5SparseMoeBlock {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let text = &cfg.text_config;
        let gate = linear_no_bias(text.hidden_size, text.num_experts, vb.pp("gate"))?;
        let shared_expert_gate = linear_no_bias(text.hidden_size, 1, vb.pp("shared_expert_gate"))?;
        let experts = (0..text.num_experts)
            .map(|idx| {
                Qwen3_5MLP::new_sized(cfg, text.moe_intermediate_size, vb.pp("experts").pp(idx))
            })
            .collect::<Result<Vec<_>>>()?;
        let shared_expert = Qwen3_5MLP::new_sized(
            cfg,
            text.shared_expert_intermediate_size,
            vb.pp("shared_expert"),
        )?;
        Ok(Self {
            gate: ProjectionImpl::Dense(gate),
            experts,
            shared_expert,
            shared_expert_gate: ProjectionImpl::Dense(shared_expert_gate),
            num_experts_per_tok: text.num_experts_per_tok,
        })
    }

    fn new_with_projections<F>(cfg: &Config, vb: VarBuilder, projections: &F) -> Result<Self>
    where
        F: for<'a> Fn(&str, VarBuilder<'a>, usize, usize, bool) -> Result<Box<dyn Projection>>,
    {
        let text = &cfg.text_config;
        let gate = projections(
            "gate",
            vb.clone(),
            text.hidden_size,
            text.num_experts,
            false,
        )?;
        let shared_expert_gate =
            projections("shared_expert_gate", vb.clone(), text.hidden_size, 1, false)?;
        let experts = (0..text.num_experts)
            .map(|idx| {
                Qwen3_5MLP::new_with_projections_sized(
                    cfg,
                    text.moe_intermediate_size,
                    vb.clone(),
                    &format!("experts.{idx}"),
                    projections,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let shared_expert = Qwen3_5MLP::new_with_projections_sized(
            cfg,
            text.shared_expert_intermediate_size,
            vb,
            "shared_expert",
            projections,
        )?;
        Ok(Self {
            gate: ProjectionImpl::Custom(Arc::from(gate)),
            experts,
            shared_expert,
            shared_expert_gate: ProjectionImpl::Custom(Arc::from(shared_expert_gate)),
            num_experts_per_tok: text.num_experts_per_tok,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = xs.dims3()?;
        let xs = xs.reshape(((), hidden_size))?;
        let weights = candle_nn::ops::softmax_last_dim(&self.gate.forward(&xs)?)?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()?;
        let mut token_indices = vec![Vec::new(); self.experts.len()];
        let mut expert_weights = vec![Vec::new(); self.experts.len()];
        for (token, scores) in weights.iter().enumerate() {
            let mut experts: Vec<_> = scores.iter().copied().enumerate().collect();
            // total_cmp makes the ordering deterministic; the secondary index breaks ties.
            experts.sort_by(|(left_idx, left), (right_idx, right)| {
                right.total_cmp(left).then_with(|| left_idx.cmp(right_idx))
            });
            let selected = &experts[..self.num_experts_per_tok.min(experts.len())];
            let total = selected.iter().map(|(_, weight)| weight).sum::<f32>();
            for &(expert, weight) in selected {
                token_indices[expert].push(token as u32);
                expert_weights[expert].push(weight / total);
            }
        }
        let mut output = xs.zeros_like()?;
        for (expert_idx, expert) in self.experts.iter().enumerate() {
            if token_indices[expert_idx].is_empty() {
                continue;
            }
            let indices = Tensor::new(token_indices[expert_idx].as_slice(), xs.device())?;
            let weights = Tensor::new(expert_weights[expert_idx].as_slice(), xs.device())?
                .reshape(((), 1))?
                .to_dtype(xs.dtype())?;
            let expert_output = expert
                .forward(&xs.index_select(&indices, 0)?)?
                .broadcast_mul(&weights)?;
            output = output.index_add(&indices, &expert_output, 0)?;
        }
        let shared_output =
            self.shared_expert
                .forward(&xs)?
                .broadcast_mul(&cuda_safe_sigmoid(
                    &self.shared_expert_gate.forward(&xs)?,
                )?)?;
        (output + shared_output)?.reshape((batch_size, seq_len, hidden_size))
    }
}

#[derive(Debug, Clone)]
enum FeedForward {
    Dense(Qwen3_5MLP),
    Sparse(Qwen3_5SparseMoeBlock),
}

impl FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Dense(mlp) => mlp.forward(xs),
            Self::Sparse(moe) => moe.forward(xs),
        }
    }
}

#[derive(Debug, Clone)]
enum TokenMixer {
    FullAttention(Qwen3_5Attention),
    LinearAttention(Qwen3_5GatedDeltaNet),
}

#[derive(Debug, Clone)]
struct Qwen3_5DecoderLayer {
    token_mixer: TokenMixer,
    mlp: FeedForward,
    input_layernorm: Qwen3_5RmsNorm,
    post_attention_layernorm: Qwen3_5RmsNorm,
}

impl Qwen3_5DecoderLayer {
    fn new(
        cfg: &Config,
        layer_idx: usize,
        rotary_emb: Arc<Qwen3_5TextRotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let layer_type = cfg
            .text_config
            .layer_types
            .get(layer_idx)
            .cloned()
            .unwrap_or_else(|| "full_attention".to_string());

        let token_mixer = if layer_type == "linear_attention" {
            TokenMixer::LinearAttention(Qwen3_5GatedDeltaNet::new(cfg, vb.pp("linear_attn"))?)
        } else {
            TokenMixer::FullAttention(Qwen3_5Attention::new(cfg, rotary_emb, vb.pp("self_attn"))?)
        };

        let mlp = if cfg.text_config.num_experts > 0 {
            FeedForward::Sparse(Qwen3_5SparseMoeBlock::new(cfg, vb.pp("mlp"))?)
        } else {
            FeedForward::Dense(Qwen3_5MLP::new(cfg, vb.pp("mlp"))?)
        };
        let input_layernorm = Qwen3_5RmsNorm::new(
            cfg.text_config.hidden_size,
            cfg.text_config.rms_norm_eps,
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = Qwen3_5RmsNorm::new(
            cfg.text_config.hidden_size,
            cfg.text_config.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            token_mixer,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn new_with_projections<F>(
        cfg: &Config,
        layer_idx: usize,
        rotary_emb: Arc<Qwen3_5TextRotaryEmbedding>,
        vb: VarBuilder,
        projections: &F,
    ) -> Result<Self>
    where
        F: for<'a> Fn(&str, VarBuilder<'a>, usize, usize, bool) -> Result<Box<dyn Projection>>,
    {
        let layer_type = cfg
            .text_config
            .layer_types
            .get(layer_idx)
            .map(String::as_str)
            .unwrap_or("full_attention");
        let token_mixer = if layer_type == "linear_attention" {
            TokenMixer::LinearAttention(Qwen3_5GatedDeltaNet::new_with_projections(
                cfg,
                vb.pp("linear_attn"),
                projections,
            )?)
        } else {
            TokenMixer::FullAttention(Qwen3_5Attention::new_with_projections(
                cfg,
                rotary_emb,
                vb.pp("self_attn"),
                projections,
            )?)
        };
        let mlp = if cfg.text_config.num_experts > 0 {
            FeedForward::Sparse(Qwen3_5SparseMoeBlock::new_with_projections(
                cfg,
                vb.pp("mlp"),
                projections,
            )?)
        } else {
            FeedForward::Dense(Qwen3_5MLP::new_with_projections(
                cfg,
                vb.pp("mlp"),
                projections,
            )?)
        };
        let input_layernorm = Qwen3_5RmsNorm::new(
            cfg.text_config.hidden_size,
            cfg.text_config.rms_norm_eps,
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = Qwen3_5RmsNorm::new(
            cfg.text_config.hidden_size,
            cfg.text_config.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            token_mixer,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &mut self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let dev = hidden_states.device().clone();
        let residual = hidden_states;
        let norm_hidden_states =
            timed("layer_norms", &dev, || self.input_layernorm.forward(hidden_states))?;

        let mixed_states = match &mut self.token_mixer {
            TokenMixer::FullAttention(attn) => timed("full_attention", &dev, || {
                attn.forward(&norm_hidden_states, attention_mask, seqlen_offset)
            })?,
            TokenMixer::LinearAttention(attn) => {
                timed("linear_attention", &dev, || attn.forward(&norm_hidden_states))?
            }
        };

        let hidden_states = (residual + mixed_states)?;
        let residual = &hidden_states;
        let norm_hidden_states =
            timed("layer_norms", &dev, || self.post_attention_layernorm.forward(&hidden_states))?;
        let mlp_states = timed("mlp", &dev, || self.mlp.forward(&norm_hidden_states))?;
        residual + mlp_states
    }

    fn clear_kv_cache(&mut self) {
        match &mut self.token_mixer {
            TokenMixer::FullAttention(attn) => attn.kv_cache = None,
            TokenMixer::LinearAttention(attn) => {
                attn.conv_state = None;
                attn.recurrent_state = None;
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<Qwen3_5DecoderLayer>,
    norm: Qwen3_5RmsNorm,
    device: Device,
    dtype: DType,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model").pp("language_model");
        let embed_tokens = candle_nn::embedding(
            cfg.text_config.vocab_size,
            cfg.text_config.hidden_size,
            vb_m.pp("embed_tokens"),
        )?;
        let rotary_emb = Arc::new(Qwen3_5TextRotaryEmbedding::new(
            vb.dtype(),
            cfg,
            vb_m.device(),
        )?);

        let mut layers = Vec::with_capacity(cfg.text_config.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.text_config.num_hidden_layers {
            let layer =
                Qwen3_5DecoderLayer::new(cfg, layer_idx, rotary_emb.clone(), vb_l.pp(layer_idx))?;
            layers.push(layer);
        }

        let norm = Qwen3_5RmsNorm::new(
            cfg.text_config.hidden_size,
            cfg.text_config.rms_norm_eps,
            vb_m.pp("norm"),
        )?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Loads Qwen 3.5 while replacing every layer projection with the result of
    /// `projections`. The factory receives the projection name, a `VarBuilder`
    /// rooted at its containing attention, linear-attention, or MLP module, and
    /// its input and output dimensions, and whether the dense projection has a
    /// bias.
    pub fn new_with_projections<F>(cfg: &Config, vb: VarBuilder, projections: F) -> Result<Self>
    where
        F: for<'a> Fn(&str, VarBuilder<'a>, usize, usize, bool) -> Result<Box<dyn Projection>>,
    {
        let vb_m = vb.pp("model").pp("language_model");
        let embed_tokens = candle_nn::embedding(
            cfg.text_config.vocab_size,
            cfg.text_config.hidden_size,
            vb_m.pp("embed_tokens"),
        )?;
        let rotary_emb = Arc::new(Qwen3_5TextRotaryEmbedding::new(
            vb.dtype(),
            cfg,
            vb_m.device(),
        )?);
        let vb_l = vb_m.pp("layers");
        let layers = (0..cfg.text_config.num_hidden_layers)
            .map(|layer_idx| {
                Qwen3_5DecoderLayer::new_with_projections(
                    cfg,
                    layer_idx,
                    rotary_emb.clone(),
                    vb_l.pp(layer_idx),
                    &projections,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let norm = Qwen3_5RmsNorm::new(
            cfg.text_config.hidden_size,
            cfg.text_config.rms_norm_eps,
            vb_m.pp("norm"),
        )?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn prepare_causal_attention_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0. }))
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), self.dtype, &self.device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
            .to_dtype(self.dtype)
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(self.prepare_causal_attention_mask(b_size, seq_len, seqlen_offset)?)
        };

        let mut xs = Module::forward(&self.embed_tokens, input_ids)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attention_mask.as_ref(), seqlen_offset)?;
        }
        self.norm.forward(&xs)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelForCausalLM {
    pub base_model: Model,
    lm_head: ProjectionImpl,
}

impl ModelForCausalLM {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let base_model = Model::new(cfg, vb.clone())?;
        let lm_head = if vb.contains_tensor("lm_head.weight") {
            linear_no_bias(
                cfg.text_config.hidden_size,
                cfg.text_config.vocab_size,
                vb.pp("lm_head"),
            )?
        } else {
            Linear::from_weights(base_model.embed_tokens.embeddings().clone(), None)
        };
        Ok(Self {
            base_model,
            lm_head: ProjectionImpl::Dense(lm_head),
        })
    }

    /// Loads Qwen 3.5 with custom layer and language-model projections.
    ///
    /// The factory receives `true` for the four attention projections when
    /// `attention_bias` is enabled, and `false` for all other projections. It
    /// is also called for `lm_head` when that tensor is present; otherwise the
    /// language model head remains tied to the input embeddings.
    pub fn new_with_projections<F>(cfg: &Config, vb: VarBuilder, projections: F) -> Result<Self>
    where
        F: for<'a> Fn(&str, VarBuilder<'a>, usize, usize, bool) -> Result<Box<dyn Projection>>,
    {
        let base_model = Model::new_with_projections(cfg, vb.clone(), &projections)?;
        let lm_head = if vb.contains_tensor("lm_head.weight") {
            ProjectionImpl::Custom(Arc::from(projections(
                "lm_head",
                vb.clone(),
                cfg.text_config.hidden_size,
                cfg.text_config.vocab_size,
                false,
            )?))
        } else {
            ProjectionImpl::Dense(Linear::from_weights(
                base_model.embed_tokens.embeddings().clone(),
                None,
            ))
        };
        Ok(Self {
            base_model,
            lm_head,
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (_b_size, seq_len) = input_ids.dims2()?;
        let xs = self
            .base_model
            .forward(input_ids, seqlen_offset)?
            .narrow(1, seq_len - 1, 1)?;
        self.lm_head.forward(&xs)
    }

    pub fn clear_kv_cache(&mut self) {
        self.base_model.clear_kv_cache();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    #[derive(Debug)]
    struct ZeroProjection {
        out_dim: usize,
        calls: Arc<AtomicUsize>,
    }

    impl Projection for ZeroProjection {
        fn forward(&self, xs: &Tensor) -> Result<Tensor> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let (tokens, _) = xs.dims2()?;
            Tensor::zeros((tokens, self.out_dim), xs.dtype(), xs.device())
        }
    }

    fn zero_mlp(hidden: usize, intermediate: usize, calls: Arc<AtomicUsize>) -> Qwen3_5MLP {
        let projection = |out_dim| {
            ProjectionImpl::Custom(Arc::new(ZeroProjection {
                out_dim,
                calls: Arc::clone(&calls),
            }))
        };
        Qwen3_5MLP {
            gate_proj: projection(intermediate),
            up_proj: projection(intermediate),
            down_proj: projection(hidden),
            act_fn: Activation::Silu,
        }
    }

    fn tiny_config(attention_bias: bool) -> Config {
        Config {
            text_config: TextConfig {
                vocab_size: 8,
                hidden_size: 4,
                intermediate_size: 6,
                num_hidden_layers: 2,
                num_attention_heads: 2,
                num_key_value_heads: 1,
                head_dim: Some(2),
                max_position_embeddings: 8,
                rms_norm_eps: 1e-5,
                rope_parameters: RopeParameters {
                    rope_theta: 10_000.,
                },
                attention_bias,
                hidden_act: Activation::Silu,
                num_experts: 0,
                num_experts_per_tok: 0,
                moe_intermediate_size: 0,
                shared_expert_intermediate_size: 0,
                partial_rotary_factor: 1.,
                layer_types: vec!["full_attention".into(), "linear_attention".into()],
                linear_num_key_heads: 1,
                linear_num_value_heads: 1,
                linear_key_head_dim: 2,
                linear_value_head_dim: 2,
                linear_conv_kernel_dim: 2,
            },
        }
    }

    #[test]
    fn projection_factory_matches_dense_model() -> Result<()> {
        for attention_bias in [false, true] {
            let device = Device::Cpu;
            let cfg = tiny_config(attention_bias);
            // Both constructors draw from the same VarMap, as they would when
            // loading a checkpoint, so this isolates the projection seam.
            let varmap = candle_nn::VarMap::new();
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
            let mut dense = Model::new(&cfg, vb.clone())?;
            let seen = Arc::new(Mutex::new(Vec::new()));
            let factory_seen = Arc::clone(&seen);
            let mut custom = Model::new_with_projections(
                &cfg,
                vb,
                move |name, vb, in_dim, out_dim, has_bias| {
                    factory_seen
                        .lock()
                        .unwrap()
                        .push((name.to_owned(), has_bias));
                    Ok(Box::new(linear_b(in_dim, out_dim, has_bias, vb.pp(name))?))
                },
            )?;

            assert_eq!(
                *seen.lock().unwrap(),
                vec![
                    ("q_proj".into(), attention_bias),
                    ("k_proj".into(), attention_bias),
                    ("v_proj".into(), attention_bias),
                    ("o_proj".into(), attention_bias),
                    ("gate_proj".into(), false),
                    ("up_proj".into(), false),
                    ("down_proj".into(), false),
                    ("in_proj_qkv".into(), false),
                    ("in_proj_z".into(), false),
                    ("in_proj_b".into(), false),
                    ("in_proj_a".into(), false),
                    ("out_proj".into(), false),
                    ("gate_proj".into(), false),
                    ("up_proj".into(), false),
                    ("down_proj".into(), false),
                ]
            );
            let tokens = Tensor::from_vec(vec![1u32, 2, 3], (1, 3), &device)?;
            assert_eq!(
                dense.forward(&tokens, 0)?.to_vec3::<f32>()?,
                custom.forward(&tokens, 0)?.to_vec3::<f32>()?,
            );
        }
        Ok(())
    }

    #[test]
    fn projection_factory_includes_untied_lm_head() -> Result<()> {
        let device = Device::Cpu;
        let cfg = tiny_config(true);
        let varmap = candle_nn::VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        // Materialise an untied output head before either constructor checks
        // whether the checkpoint provides it.
        let _ = vb.get(
            (cfg.text_config.vocab_size, cfg.text_config.hidden_size),
            "lm_head.weight",
        )?;
        let mut dense = ModelForCausalLM::new(&cfg, vb.clone())?;
        let seen = Arc::new(Mutex::new(Vec::new()));
        let factory_seen = Arc::clone(&seen);
        let mut custom = ModelForCausalLM::new_with_projections(
            &cfg,
            vb,
            move |name, vb, in_dim, out_dim, has_bias| {
                factory_seen.lock().unwrap().push(name.to_owned());
                Ok(Box::new(linear_b(in_dim, out_dim, has_bias, vb.pp(name))?))
            },
        )?;

        assert_eq!(seen.lock().unwrap().last(), Some(&"lm_head".to_owned()));
        let tokens = Tensor::from_vec(vec![1u32, 2, 3], (1, 3), &device)?;
        assert_eq!(
            dense.forward(&tokens, 0)?.to_vec3::<f32>()?,
            custom.forward(&tokens, 0)?.to_vec3::<f32>()?,
        );
        Ok(())
    }

    #[test]
    fn partial_rotary_keeps_the_unrotated_tail() -> Result<()> {
        let device = Device::Cpu;
        let mut cfg = tiny_config(false);
        cfg.text_config.head_dim = Some(4);
        cfg.text_config.partial_rotary_factor = 0.5;
        let rotary = Qwen3_5TextRotaryEmbedding::new(DType::F32, &cfg, &device)?;
        let q = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 1, 1, 4), &device)?;
        let (q, _) = rotary.apply(&q, &q, 1)?;
        assert_eq!(&q.flatten_all()?.to_vec1::<f32>()?[2..], &[3., 4.]);
        Ok(())
    }

    #[test]
    fn sparse_moe_groups_tokens_and_breaks_router_ties_by_expert_index() -> Result<()> {
        let device = Device::Cpu;
        let calls = [Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0))];
        let hidden = 2;
        let moe = Qwen3_5SparseMoeBlock {
            // Equal router probabilities must select expert zero first.
            gate: ProjectionImpl::Custom(Arc::new(ZeroProjection {
                out_dim: 2,
                calls: Arc::new(AtomicUsize::new(0)),
            })),
            experts: vec![
                zero_mlp(hidden, 3, Arc::clone(&calls[0])),
                zero_mlp(hidden, 3, Arc::clone(&calls[1])),
            ],
            shared_expert: zero_mlp(hidden, 3, Arc::new(AtomicUsize::new(0))),
            shared_expert_gate: ProjectionImpl::Custom(Arc::new(ZeroProjection {
                out_dim: 1,
                calls: Arc::new(AtomicUsize::new(0)),
            })),
            num_experts_per_tok: 1,
        };
        let xs = Tensor::zeros((1, 4, hidden), DType::F32, &device)?;
        moe.forward(&xs)?;
        // Each MLP projection runs once for the whole chunk, not once per token.
        assert_eq!(calls[0].load(Ordering::Relaxed), 3);
        assert_eq!(calls[1].load(Ordering::Relaxed), 0);
        Ok(())
    }
}
