// V2-Q2 isolated bidirectional graded-deviation kernel.
// The retained graph and all Q1 dynamics are preserved.  The only changed
// mechanism is the signed [-1, 1] release-deviation clamp.
#include <algorithm>
#include <cmath>
#include <cstdint>

extern "C" void v2_q2_advance(
    int n, const int64_t* ptr, const int32_t* post, const float* weight,
    float* v, float* g_event, float* g_graded, int16_t* refractory,
    const float* drive, float* previous_drive, int32_t* spike_queue,
    int32_t* spike_queue_count, float* graded_queue, float* graded_drive,
    int64_t* clock, int steps, float dt, const uint8_t* graded_mask,
    const uint8_t* kc_mask, const float* rest, float* adaptation,
    float adaptation_jump, float adaptation_tau, int32_t* counts,
    int32_t* active, uint8_t* active_flag, int32_t* nactive,
    int64_t* last, double* eligibility, int64_t* eligibility_last,
    float* modulation, int64_t* modulation_last,
    const uint8_t* modulation_mask, double* release_integral,
    double* signed_delivery, double* absolute_delivery, int32_t* overflow,
    double* edge_signed_delivery, int64_t* edge_event_count,
    int64_t* edge_first_arrival_clock) {
  const int delay = (int)std::lround(1.8f / dt);
  const int rfc = (int)std::lround(2.2f / dt);
  const int slots = delay + 1;
  const float tau_m = 20.f, tau_s = 5.f, tau_emit = 2.2f;
  const float threshold = -45.f;
  const float a = std::exp(-dt / tau_m);
  const float b = std::exp(-dt / tau_s);

  auto awaken = [&](int i) {
    if (!active_flag[i]) {
      if (*nactive >= n) { *overflow = 1; return; }
      active_flag[i] = 1;
      active[(*nactive)++] = i;
    }
  };
  auto evolve = [&](int i, bool was_ref) {
    const float u = graded_drive[i];
    const float gss = u * tau_s;
    const float oldg = g_graded[i];
    g_graded[i] = gss + (oldg - gss) * b;
    if (was_ref) {
      g_event[i] *= b;
    } else {
      const float oldv = v[i];
      const float ge = g_event[i];
      v[i] = rest[i] + drive[i] + gss
           + (oldv - rest[i] - drive[i] - gss) * a
           + (oldg - gss) * (tau_s / (tau_m - tau_s)) * (a - b)
           + ge * (a - b) / 3.f;
      g_event[i] = ge * b;
      if (kc_mask[i]) {
        const float c = std::exp(-dt / 200.f);
        v[i] -= adaptation[i] * 200.f / (200.f - tau_m) * (c - a);
        adaptation[i] *= c;
      }
    }
  };

  for (int i = 0; i < n; ++i) {
    if (drive[i] != previous_drive[i]) {
      previous_drive[i] = drive[i];
      awaken(i);
    }
  }
  for (int t = 0; t < steps; ++t, ++(*clock)) {
    const int slot = (int)(*clock % slots);
    const int future = (int)((*clock + delay) % slots);
    const int original = *nactive;
    int kept = 0;
    for (int k = 0; k < original; ++k) {
      const int i = active[k];
      const bool was_ref = refractory[i] > 0;
      if (was_ref) --refractory[i];
      evolve(i, was_ref);
      last[i] = *clock;
      if (!graded_mask[i] && refractory[i] == 0 && v[i] > threshold) {
        if (spike_queue_count[future] >= n) { *overflow = 1; continue; }
        spike_queue[future * n + spike_queue_count[future]++] = i;
        counts[i] += 1;
        if (kc_mask[i]) {
          adaptation[i] += adaptation_jump;
          const double elapsed = dt * (double)(*clock - eligibility_last[i]);
          eligibility[i] *= std::exp(-elapsed / 100.0);
          eligibility[i] += 1.0;
          eligibility_last[i] = *clock;
        }
      }
      if (graded_mask[i]) {
        float r = (v[i] - rest[i]) / (threshold - rest[i]);
        r = std::max(-1.f, std::min(1.f, r));
        *release_integral += (double)r * (double)dt;
        if (r != 0.f) {
          for (int64_t e = ptr[i]; e < ptr[i + 1]; ++e) {
            const int j = post[e];
            const double q = (double)weight[e] * (double)r / tau_emit;
            graded_queue[future * n + j] += (float)q;
            *signed_delivery += q * (double)dt;
            *absolute_delivery += std::abs(q * (double)dt);
            if (edge_signed_delivery) {
              edge_signed_delivery[e] += q * (double)dt;
              edge_event_count[e] += 1;
              if (edge_first_arrival_clock[e] < 0)
                edge_first_arrival_clock[e] = *clock + delay;
            }
          }
        }
      }
      const bool live = std::fabs(drive[i]) > 1e-7f ||
                        std::fabs(g_event[i]) > 1e-7f ||
                        std::fabs(g_graded[i]) > 1e-7f ||
                        std::fabs(v[i] - rest[i]) > 1e-7f ||
                        v[i] > threshold || drive[i] > threshold - rest[i];
      if (live || graded_mask[i]) active[kept++] = i;
      else active_flag[i] = 0;
    }
    *nactive = kept;

    for (int q = 0; q < spike_queue_count[slot]; ++q) {
      const int source = spike_queue[slot * n + q];
      for (int64_t e = ptr[source]; e < ptr[source + 1]; ++e) {
        const int j = post[e];
        if (modulation_mask[source]) {
          const double elapsed = dt * (double)(*clock - modulation_last[j]);
          modulation[j] *= (float)std::exp(-elapsed / 100.0);
          modulation[j] += std::fabs(weight[e]) / 0.275f;
          modulation_last[j] = *clock;
        } else if (graded_mask[j] || refractory[j] == 0) {
          g_event[j] += weight[e];
          if (edge_signed_delivery) {
            edge_signed_delivery[e] += (double)weight[e];
            edge_event_count[e] += 1;
            if (edge_first_arrival_clock[e] < 0)
              edge_first_arrival_clock[e] = *clock;
          }
          awaken(j);
        }
      }
    }
    spike_queue_count[slot] = 0;
    for (int j = 0; j < n; ++j) {
      const float incoming = graded_queue[slot * n + j];
      graded_drive[j] = incoming;
      graded_queue[slot * n + j] = 0.f;
      if (incoming != 0.f) awaken(j);
    }
    for (int q = 0; q < spike_queue_count[future]; ++q) {
      const int i = spike_queue[future * n + q];
      v[i] = rest[i];
      g_event[i] = 0.f;
      refractory[i] = (int16_t)rfc;
      awaken(i);
    }
  }
}

// Immutable v1 compatibility implementation under a Q2-private symbol.
#define memory_advance q2_embedded_memory_advance
#include "kernel.cpp"
#undef memory_advance

extern "C" void v2_q2_compat_advance(
 int n,const int64_t* ptr,const int32_t* post,float* weight,float* v,float* g,
 int16_t* refractory,const float* drive,float* previous_drive,int32_t* queue,
 int32_t* queue_count,int64_t* clock,int steps,float dt,int32_t* counts,
 int32_t* active,uint8_t* flags,int32_t* nactive,int64_t* last,
 const uint8_t* kc_mask,const int8_t* dan_index,double* eligibility,
 int64_t* eligibility_last,int nplastic,const int64_t* plastic_edge,
 const int32_t* plastic_pre,const float* baseline_weight,const float* dan_gain,
 float eta,float tau_elig_ms,float floor_fraction,int learning_enabled,
 float* modulation,int64_t* modulation_last,const uint8_t* modulation_mask,
 const float* rest,float* adaptation,float adaptation_jump,float adaptation_tau,
 void* t1,void* t2,void* t3,void* t4,void* t5,void* t6,void* t7,void* t8) {
  q2_embedded_memory_advance(n,ptr,post,weight,v,g,refractory,drive,
    previous_drive,queue,queue_count,clock,steps,dt,counts,active,flags,nactive,
    last,kc_mask,dan_index,eligibility,eligibility_last,nplastic,plastic_edge,
    plastic_pre,baseline_weight,dan_gain,eta,tau_elig_ms,floor_fraction,
    learning_enabled,modulation,modulation_last,modulation_mask,rest,adaptation,
    adaptation_jump,adaptation_tau,t1,t2,t3,t4,t5,t6,t7,t8);
}
