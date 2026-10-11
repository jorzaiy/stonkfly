// V2-Q1 isolated hybrid kernel.
//
// This translation unit intentionally has no dependency on the v1 kernel or
// on market, reward, plasticity, controller, or decoder code.  Graded nodes
// use a continuous release queue; all other nodes retain the v1 event path.
#include <cmath>
#include <cstdint>
#include <algorithm>

extern "C" void v2_hybrid_advance(
    int n, const int64_t* ptr, const int32_t* post, const float* weight,
    float* v, float* g_event, float* g_graded, int16_t* refractory,
    const float* drive, float* previous_drive, int32_t* spike_queue,
    int32_t* spike_queue_count, float* graded_queue, float* graded_drive,
    int64_t* clock, int steps, float dt, const uint8_t* graded_mask,
    const uint8_t* kc_mask, const float* rest, float* adaptation,
    float adaptation_jump, float adaptation_tau, int32_t* counts,
    int32_t* active, uint8_t* active_flag, int32_t* nactive,
    double* release_integral, double* signed_delivery,
    double* absolute_delivery, int32_t* overflow) {
  const int delay = std::lround(1.8f / dt);
  const int rfc = std::lround(2.2f / dt);
  const int slots = delay + 1;
  const float tau_m = 20.f, tau_s = 5.f, tau_emit = 2.2f;
  const float threshold = -45.f;
  const float inv_tau_m = 1.f / tau_m;
  const float inv_tau_s = 1.f / tau_s;
  auto expm = [&](float x) { return std::exp(-x); };
  auto awaken = [&](int i) {
    if (!active_flag[i]) {
      if (*nactive >= n) { *overflow = 1; return; }
      active_flag[i] = 1;
      active[(*nactive)++] = i;
    }
  };
  // Analytic segment update. Event and graded conductances are deliberately
  // separate; graded conductance is updated even while a spiking target is
  // refractory, while its voltage retains the v1 clamp/reset semantics.
  auto evolve = [&](int i, int64_t now, float current) {
    int64_t d = now - (int64_t)0; // replaced below by per-node last via clock-local schedule
    (void)d;
  };
  // v1's observation schedule materializes every active node once per step;
  // the local last clock is represented by a compact side channel in the
  // queue count's unused negative range is not acceptable, so this kernel
  // integrates one dt per active node per call.  Drive changes are settled at
  // the call boundary by one exact segment, matching v1's public step API.
  for (int i = 0; i < n; ++i) {
    if (drive[i] != previous_drive[i]) awaken(i);
    previous_drive[i] = drive[i];
  }
  for (int t = 0; t < steps; ++t, ++(*clock)) {
    const int slot = (int)(*clock % slots);
    const int future = (int)((*clock + delay) % slots);
    const int original = *nactive;
    int kept = 0;
    const float a = expm(dt * inv_tau_m), b = expm(dt * inv_tau_s);
    const float adapt_a = expm(dt / adaptation_tau);
    for (int k = 0; k < original; ++k) {
      const int i = active[k];
      const bool in_ref = refractory[i] > 0;
      if (in_ref) {
        refractory[i] = refractory[i] > 0 ? (int16_t)(refractory[i] - 1) : 0;
      }
      const float u = graded_drive[i];
      const float gss = u * tau_s;
      // Conductance updates are done for every active node.  A refractory
      // spiking target keeps its voltage fixed, but its graded state evolves.
      const float oldg = g_graded[i];
      g_graded[i] = gss + (oldg - gss) * b;
      if (!in_ref) {
        const float oldv = v[i];
        const float ge = g_event[i];
        v[i] = rest[i] + drive[i] + gss
             + (oldv - rest[i] - drive[i] - gss) * a
             + (oldg - gss) * (tau_s / (tau_m - tau_s)) * (a - b)
             + ge * (a - b) / 3.f;
        g_event[i] = ge * b;
        if (kc_mask[i]) adaptation[i] *= adapt_a;
      } else {
        g_event[i] *= b;
        if (kc_mask[i]) adaptation[i] *= adapt_a;
      }
      if (!graded_mask[i] && refractory[i] == 0 && v[i] > threshold) {
        if (spike_queue_count[future] >= n) { *overflow = 1; continue; }
        spike_queue[future * n + spike_queue_count[future]++] = i;
        counts[i] += 1;
        if (kc_mask[i]) adaptation[i] += adaptation_jump;
      }
      if (graded_mask[i]) {
        float r = (v[i] - rest[i]) / (threshold - rest[i]);
        r = std::max(0.f, std::min(1.f, r));
        *release_integral += (double)r * (double)dt;
        for (int64_t e = ptr[i]; e < ptr[i + 1]; ++e) {
          const int j = post[e];
          const float q = weight[e] * r * dt / tau_emit;
          graded_queue[future * n + j] += q / dt; // queue stores rate
          *signed_delivery += (double)q;
          *absolute_delivery += std::abs((double)q);
        }
      }
      const bool graded_live = graded_mask[i] &&
          (std::fabs(drive[i]) > 1e-7f || std::fabs(g_event[i]) > 1e-7f ||
           std::fabs(g_graded[i]) > 1e-7f || std::fabs(v[i] - rest[i]) > 1e-7f);
      const bool can_fire = graded_live || std::fabs(g_event[i]) > 1e-7f ||
                            std::fabs(g_graded[i]) > 1e-7f || v[i] > threshold ||
                            drive[i] > threshold - rest[i] ||
                            drive[i] + g_event[i] + g_graded[i] > threshold - rest[i];
      if (can_fire) active[kept++] = i; else active_flag[i] = 0;
    }
    *nactive = kept;

    // Current-slot discrete events: spiking→graded is never dropped; for a
    // spiking target v1's refractory drop is retained.
    for (int q = 0; q < spike_queue_count[slot]; ++q) {
      const int source = spike_queue[slot * n + q];
      for (int64_t e = ptr[source]; e < ptr[source + 1]; ++e) {
        const int j = post[e];
        if (graded_mask[j] || refractory[j] == 0) {
          g_event[j] += weight[e];
          awaken(j);
        }
      }
    }
    spike_queue_count[slot] = 0;
    // Apply arrivals after the current state segment; they therefore affect
    // the next interval exactly as specified by the Q1 step order.
    for (int j = 0; j < n; ++j) {
      const float incoming = graded_queue[slot * n + j];
      graded_drive[j] = incoming;
      graded_queue[slot * n + j] = 0.f;
      if (incoming != 0.f) awaken(j);
    }
    // A queued spike reset is an event reset only: g_graded is preserved.
    for (int q = 0; q < spike_queue_count[future]; ++q) {
      const int i = spike_queue[future * n + q];
      v[i] = rest[i];
      g_event[i] = 0.f;
      refractory[i] = (int16_t)rfc;
      awaken(i);
    }
  }
}
