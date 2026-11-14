// Gated delta rule (GatedDeltaNet) recurrence over a whole prompt, one block per (batch, head).
//
// For every token t, with state S of shape [d_k, d_v] (per head):
//   S      = S * exp(log_g_t)
//   kv_mem = k_t^T S                      -> [d_v]
//   delta  = (v_t - kv_mem) * beta_t
//   S      = S + k_t delta^T
//   out_t  = q_t^T S                      -> [d_v]
//
// Each thread owns one column v of S in registers, so the state never leaves the SM during the
// prompt. q_t and k_t are shared by the block through shared memory. q is expected to be
// already scaled by 1/sqrt(d_k) and q, k L2-normalised, as the model layer does.
//
// Layouts (all contiguous, F32):
//   qkv    [b, t, h, 3 * d]   q, k, v of each (token, head) packed one after the other
//   gb     [b, t, h, 2]       log decay and beta
//   state  [b, h, d, d]       S[k][v], v fastest
//   out    [b * t * h * d  +  b * h * d * d]   outputs, then the final state
// Requires d_k == d_v == d <= GDN_MAX_DIM, with blockDim.x == d.

#define GDN_MAX_DIM 128

extern "C" __global__ void gated_delta_prefill_f32(
    const float* __restrict__ qkv,
    const float* __restrict__ gb,
    const float* __restrict__ state_in,
    float* __restrict__ out,
    const unsigned int b,
    const unsigned int t,
    const unsigned int h,
    const unsigned int d
) {
    __shared__ float q_s[GDN_MAX_DIM];
    __shared__ float k_s[GDN_MAX_DIM];

    const unsigned int bh = blockIdx.x;
    const unsigned int bi = bh / h;
    const unsigned int hi = bh % h;
    const unsigned int v = threadIdx.x;

    float s[GDN_MAX_DIM];
    const float* state_head = state_in + (size_t)bh * d * d;
#pragma unroll
    for (int k = 0; k < GDN_MAX_DIM; ++k) {
        s[k] = (k < (int)d) ? state_head[(size_t)k * d + v] : 0.0f;
    }

    for (unsigned int ti = 0; ti < t; ++ti) {
        const size_t token = (size_t)bi * t + ti;
        const float* row = qkv + (token * h + hi) * 3 * d;
        const float* gate = gb + (token * h + hi) * 2;

        __syncthreads();  // the previous token is done with q_s and k_s
        q_s[v] = row[v];
        k_s[v] = row[d + v];
        const float v_t = row[2 * d + v];
        const float g = expf(gate[0]);
        const float beta = gate[1];
        __syncthreads();

        float kv_mem = 0.0f;
#pragma unroll
        for (int k = 0; k < GDN_MAX_DIM; ++k) {
            if (k < (int)d) {
                s[k] *= g;
                kv_mem += s[k] * k_s[k];
            }
        }
        const float delta = (v_t - kv_mem) * beta;
        float o = 0.0f;
#pragma unroll
        for (int k = 0; k < GDN_MAX_DIM; ++k) {
            if (k < (int)d) {
                s[k] += k_s[k] * delta;
                o += s[k] * q_s[k];
            }
        }
        out[(token * h + hi) * d + v] = o;
    }

    float* state_out = out + (size_t)b * t * h * d + (size_t)bh * d * d;
#pragma unroll
    for (int k = 0; k < GDN_MAX_DIM; ++k) {
        if (k < (int)d) state_out[(size_t)k * d + v] = s[k];
    }
}
