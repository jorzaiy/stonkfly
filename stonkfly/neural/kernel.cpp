// Experimental dopamine-gated KC-to-MBON11 depression.
// Baseline integration is copied without changing its timestep/schedule.
// PPL101 edges deliver a modeled modulatory trace instead of fast excitation.
// Lazy exact subthreshold evolution. An inactive neuron is skipped only when
// its voltage AND both its instantaneous and asymptotic drive are below the
// threshold. With no incoming event it cannot fire. Every edge is retained.
#include <cmath>
#include <cstdint>
#include <vector>
struct TraceObserver {
 int32_t steps, candidate_n, kc_n, overflow;
 const int32_t* candidate_slot; const uint8_t* cb_mask; const int32_t* kc_slot;
 int32_t* step_spike_counts; int32_t* candidate_spike_counts;
 int32_t* candidate_event_counts; double* candidate_signed; double* candidate_positive; double* candidate_negative;
 int32_t* cbkc_event_counts; double* cbkc_signed; double* cbkc_positive; double* cbkc_negative;
 int64_t* kc_first_spike_step; int64_t start_clock;
};
struct TraceObserverS2F { uint64_t magic; int32_t abi_version, size, projection_n, cb_n, kc_n, overflow, kc_unique; const int32_t* projection_slot; const int32_t* cb_slot; const int32_t* kc_slot; int32_t* kc_first; int32_t* projection_spikes; int32_t* projection_first; int32_t* projection_last; int32_t* projection_events; double* projection_signed; double* projection_positive; double* projection_negative; int32_t* cb_first; int32_t* cb_events; double* cb_signed; double* cb_positive; double* cb_negative; };
struct TraceObserverS2Fv2 { uint64_t magic; int32_t abi_version, size, projection_n, cb_n, kc_n, overflow, kc_unique; const int32_t* projection_slot; const int32_t* cb_slot; const int32_t* kc_slot; int32_t* kc_first; int32_t* projection_first_delivery; int32_t* projection_spikes; int32_t* projection_first; int32_t* projection_last; int32_t* projection_events; double* projection_signed; double* projection_positive; double* projection_negative; int32_t* cb_first; int32_t* cb_events; double* cb_signed; double* cb_positive; double* cb_negative; };
struct TraceObserverS2G { uint64_t magic; int32_t abi_version, size, source_n, target_n, overflow; const int32_t* source_slot; const int32_t* target_slot; int32_t* target_first; int32_t* pair_events; double* pair_signed; double* pair_positive; double* pair_negative; int32_t* pair_first; int32_t* pair_last; };
struct TraceObserverS2H { uint64_t magic; int32_t abi_version, size, source_n, stable_n, kc_n, target_node, out_capacity, out_used, kc_order_used, overflow; const int32_t* source_slot; const int32_t* stable_slot; const int32_t* kc_slot; int32_t* source_spikes; int32_t* source_first; int32_t* target_first; int32_t* stable_first; int32_t* kc_first; int32_t* kc_order_id; int32_t* kc_order_step; int32_t* source_events; double* source_signed; double* source_positive; double* source_negative; int32_t* source_first_delivery; int32_t* source_last_delivery; int32_t* source_pre_count; int32_t* source_pre_first; int32_t* source_pre_last; int32_t* out_step; int32_t* out_target; double* out_weight; };
struct TraceObserverS2I { uint64_t magic; int32_t abi_version, size, source_n, target_node, overflow; const int32_t* source_slot; int32_t* source_spikes; int32_t* source_first; int32_t* source_events; double* source_signed; double* source_positive; double* source_negative; int32_t* source_first_delivery; int32_t* source_last_delivery; int32_t* source_pre_count; int32_t* source_pre_first; int32_t* source_pre_last; int32_t* leader_first; };
struct TraceObserverS2L { uint64_t magic; int32_t abi_version, size, source_n, kc_n, overflow; const int8_t* source_class; const int32_t* kc_slot; int32_t* kc_first; int32_t* kc_unique; int32_t* kc1_step; int32_t* kc10_step; int32_t* kc_order_id; int32_t* kc_order_step; int32_t* kc_order_used; int64_t* phase_events; double* phase_signed; double* phase_positive; double* phase_negative; int32_t* phase_unique_source; int32_t* phase_unique_target; int32_t* phase_first; int32_t* phase_last; int64_t* all_kc_delivery_count; uint8_t* source_seen; uint8_t* target_seen; int64_t* kc_class_events; double* kc_class_signed; double* kc_class_positive; double* kc_class_negative; };
struct TraceObserverS2M { uint64_t magic; int32_t abi_version, size, source_n, kc_n, other_cb_n, fixed_kc_n, overflow; const int8_t* source_class; const int32_t* kc_slot; const int32_t* other_cb_slot; const int32_t* kc_fixed_slot; int32_t* kc_first; int32_t* kc1_step; int64_t* events; double* signed_w; double* positive; double* negative; int32_t* unique_source; int32_t* unique_target; int32_t* first; int32_t* last; uint8_t* source_seen; uint8_t* target_seen; int64_t* other_cb_events; double* other_cb_signed; double* other_cb_positive; double* other_cb_negative; int32_t* other_cb_first; int32_t* other_cb_last; uint8_t* other_cb_source_seen; int64_t* fixed_events; double* fixed_signed; double* fixed_positive; double* fixed_negative; int32_t* fixed_first; int32_t* fixed_last; int64_t* audit; };
struct TraceObserverS2N { uint64_t magic; int32_t abi_version, size, source_n, kc_n, overflow; const int8_t* source_group; const int32_t* kc_slot; const double* scale; int32_t* kc1_step; int64_t* eligible_count; int64_t* applied_count; int64_t* eligible_events; int64_t* applied_events; double* original_signed; double* original_positive; double* original_negative; double* applied_signed; double* applied_positive; double* applied_negative; int64_t* selected_outside_count; };
extern "C" void memory_advance(
 int n,const int64_t* ptr,const int32_t* post,float* weight,
 float* v,float* g,int16_t* refractory,const float* drive,float* previous_drive,
 int32_t* queue,int32_t* queue_count,int64_t* clock,int steps,float dt,int32_t* counts,
 int32_t* active,uint8_t* flags,int32_t* nactive,int64_t* last,
 const uint8_t* kc_mask,const int8_t* dan_index,double* eligibility,int64_t* eligibility_last,
 int nplastic,const int64_t* plastic_edge,const int32_t* plastic_pre,
 const float* baseline_weight,const float* dan_gain,float eta,float tau_elig_ms,float floor_fraction,
 int learning_enabled,float* modulation,int64_t* modulation_last,const uint8_t* modulation_mask,const float* rest,
 float* adaptation,float adaptation_jump,float adaptation_tau,void* trace_ptr,void* trace2_ptr,void* trace3_ptr,void* trace4_ptr,void* trace5_ptr,void* trace6_ptr,void* trace7_ptr,void* trace8_ptr) {
 TraceObserver* trace = static_cast<TraceObserver*>(trace_ptr);
 TraceObserverS2Fv2* trace2 = static_cast<TraceObserverS2Fv2*>(trace2_ptr);
 TraceObserverS2G* trace3 = static_cast<TraceObserverS2G*>(trace3_ptr);
 TraceObserverS2H* trace4 = static_cast<TraceObserverS2H*>(trace4_ptr);
 TraceObserverS2I* trace5 = static_cast<TraceObserverS2I*>(trace5_ptr);
 TraceObserverS2L* trace6 = static_cast<TraceObserverS2L*>(trace6_ptr);
 TraceObserverS2M* trace7 = static_cast<TraceObserverS2M*>(trace7_ptr);
 TraceObserverS2N* trace8 = static_cast<TraceObserverS2N*>(trace8_ptr);
 if(trace2 && (trace2->magic!=0x5332465452414346ULL || trace2->abi_version!=2 || trace2->size!=(int)sizeof(TraceObserverS2Fv2))){trace2->overflow=1; return;}
 if(trace3 && (trace3->magic!=0x5332475452414346ULL || trace3->abi_version!=1 || trace3->size!=(int)sizeof(TraceObserverS2G))){trace3->overflow=1; return;}
 if(trace4 && (trace4->magic!=0x5332485452414346ULL || trace4->abi_version!=3 || trace4->size!=(int)sizeof(TraceObserverS2H))){trace4->overflow=1; return;}
 if(trace5 && (trace5->magic!=0x5332495452414346ULL || trace5->abi_version!=1 || trace5->size!=(int)sizeof(TraceObserverS2I))){trace5->overflow=1; return;}
 if(trace6 && (trace6->magic!=0x53324c5452414346ULL || trace6->abi_version!=1 || trace6->size!=(int)sizeof(TraceObserverS2L))){trace6->overflow=1; return;}
 if(trace7 && (trace7->magic!=0x53324d5452414346ULL || trace7->abi_version!=2 || trace7->size!=(int)sizeof(TraceObserverS2M))){trace7->overflow=1; return;}
 if(trace8 && (trace8->magic!=0x53324e5452414346ULL || trace8->abi_version!=2 || trace8->size!=(int)sizeof(TraceObserverS2N))){trace8->overflow=1; return;}
 const int delay=std::lround(1.8f/dt),rfc=std::lround(2.2f/dt),slots=delay+1;
 float av[1024],ag[1024],aa[1024];
 for(int i=0;i<1024;i++){av[i]=std::exp(-dt*i/20.f);ag[i]=std::exp(-dt*i/5.f);aa[i]=std::exp(-dt*i/adaptation_tau);}
 auto evolve=[&](int i,int64_t now,float current){
   int64_t d=now-last[i];if(d<=0)return;
   const int frozen=refractory[i]>0?refractory[i]-1:0;
   const int skip=(int)(d<frozen?d:frozen);
   if(skip>0 && adaptation[i]>0)adaptation[i]*=skip<1024?aa[skip]:std::exp(-dt*skip/adaptation_tau);
   refractory[i]=d>=refractory[i]?0:refractory[i]-d;d-=skip;
   if(d>0){const float a=d<1024?av[d]:std::exp(-dt*d/20.f),b=d<1024?ag[d]:std::exp(-dt*d/5.f);
     v[i]=rest[i]+(v[i]-rest[i])*a+current*(1.f-a)+g[i]*(a-b)/3.f;g[i]*=b;
     if(adaptation[i]>0){const float c=d<1024?aa[d]:std::exp(-dt*d/adaptation_tau);
       v[i]-=adaptation[i]*adaptation_tau/(adaptation_tau-20.f)*(c-a);adaptation[i]*=c;}
   }
   last[i]=now;
 };
 auto awaken=[&](int i){if(!flags[i]){flags[i]=1;active[(*nactive)++]=i;}};
 auto observe_s2m=[&](int source,int target,int64_t edge){
   if(!trace7) return;
   const int ks=trace7->kc_slot[target],sc=trace7->source_class[source];
   if(ks<0 || trace7->kc1_step[0]<0) return;
   const int d=*clock-trace7->kc1_step[0];
   // The lower and upper bounds are both explicit: pre-KC1 and post-window
   // deliveries are intentionally outside S2m's five mutually-exclusive bins.
   if(!(d>=0 && d<1024)) return;
   if(sc<0 || sc>=6){trace7->overflow=1;return;}
   int w=-1;
   if(d>=0 && d<64) w=0;
   else if(d>=64 && d<128) w=1;
   else if(d>=128 && d<256) w=2;
   else if(d>=256 && d<512) w=3;
   else if(d>=512 && d<1024) w=4;
   if(w<0) return;
   trace7->audit[0]++;
   const int z=sc*5+w; const double q=weight[edge];
   trace7->events[z]++; trace7->signed_w[z]+=q;
   if(q>0) trace7->positive[z]+=q; else if(q<0) trace7->negative[z]+=q;
   if(!trace7->source_seen[source*5+w]){trace7->source_seen[source*5+w]=1;trace7->unique_source[z]++;}
   if(!trace7->target_seen[ks*6*5+sc*5+w]){trace7->target_seen[ks*6*5+sc*5+w]=1;trace7->unique_target[z]++;}
   if(trace7->first[z]<0) trace7->first[z]=d; trace7->last[z]=d;
   const int os=trace7->other_cb_slot[source];
   if(os>=0 && os<trace7->other_cb_n){
     const int oz=os*5+w; trace7->other_cb_events[oz]++; trace7->other_cb_signed[oz]+=q;
     if(q>0) trace7->other_cb_positive[oz]+=q; else if(q<0) trace7->other_cb_negative[oz]+=q;
     if(trace7->other_cb_first[oz]<0) trace7->other_cb_first[oz]=d; trace7->other_cb_last[oz]=d;
     trace7->other_cb_source_seen[oz]=1;
   }
   const int fs=trace7->kc_fixed_slot[ks];
   if(fs>=0 && fs<trace7->fixed_kc_n){
     const int fz=fs*6*5+sc*5+w; trace7->fixed_events[fz]++; trace7->fixed_signed[fz]+=q;
     if(q>0) trace7->fixed_positive[fz]+=q; else if(q<0) trace7->fixed_negative[fz]+=q;
     if(trace7->fixed_first[fz]<0) trace7->fixed_first[fz]=d; trace7->fixed_last[fz]=d;
   }
 };
 auto intervene_s2n=[&](int source,int target,int64_t edge){
   if(!trace8) return weight[edge];
   const int group=trace8->source_group[source];
   if(group<1 || group>2) return weight[edge];
   const int ks=trace8->kc_slot[target];
   if(ks<0 || trace8->kc1_step[0]<0) return weight[edge];
   const int d=*clock-trace8->kc1_step[0];
   if(!(d>=0 && d<1024)){trace8->selected_outside_count[0]++;return weight[edge];}
   int w=-1;
   if(d>=0 && d<64)w=0; else if(d>=64 && d<128)w=1; else if(d>=128 && d<256)w=2; else if(d>=256 && d<512)w=3; else if(d>=512 && d<1024)w=4;
   if(w<0){trace8->selected_outside_count[0]++;return weight[edge];}
   // Visual sources are eligible only in W1/W2; frozen other_cb sources
   // are eligible only in W4. BOTH is their disjoint union.
   const bool eligible=(group==1 && (w==1 || w==2)) || (group==2 && w==4);
   if(!eligible) return weight[edge];
   const int z=(group-1)*5+w;
   const double q=weight[edge], s=trace8->scale[w];
   trace8->eligible_count[0]++; trace8->eligible_events[z]++; trace8->original_signed[z]+=q;
   if(q>0)trace8->original_positive[z]+=q; else if(q<0)trace8->original_negative[z]+=q;
   if(s==1.0) return weight[edge];
   trace8->applied_count[0]++; trace8->applied_events[z]++; trace8->applied_signed[z]+=q*s;
   if(q*s>0)trace8->applied_positive[z]+=q*s; else if(q*s<0)trace8->applied_negative[z]+=q*s;
   return static_cast<float>(q*s);
 };
 // Apply changing sensory currents only after settling old-current history.
 for(int i=0;i<n;i++)if(drive[i]!=previous_drive[i]){
   evolve(i,*clock-1,previous_drive[i]);previous_drive[i]=drive[i];awaken(i);
 }
 for(int t=0;t<steps;t++,(*clock)++){
   const int slot=*clock%slots,future=(*clock+delay)%slots;
   int kept=0,original=*nactive;
   for(int k=0;k<original;k++){
     const int i=active[k];evolve(i,*clock,drive[i]);
     if(refractory[i]==0 && v[i]>-45.f){queue[future*n+queue_count[future]++]=i;counts[i]++;
       if(trace){const int64_t ts=*clock-trace->start_clock;const int cs=trace->candidate_slot[i],ks=trace->kc_slot[i];if(ts<0||ts>=trace->steps){trace->overflow=1;}else{if(cs>=0){trace->step_spike_counts[ts*3]+=1;trace->candidate_spike_counts[cs*trace->steps+ts]++;}else if(ks>=0){trace->step_spike_counts[ts*3+2]+=1;if(trace->kc_first_spike_step[ks]<0)trace->kc_first_spike_step[ks]=ts;}else if(trace->cb_mask[i])trace->step_spike_counts[ts*3+1]+=1;}}
       if(trace2){const int ps=trace2->projection_slot[i],ks=trace2->kc_slot[i];if(ks>=0&&trace2->kc_first[ks]<0){trace2->kc_first[ks]=*clock;trace2->kc_unique++;}if(ps>=0){trace2->projection_spikes[ps]++;if(trace2->projection_first[ps]<0)trace2->projection_first[ps]=*clock;trace2->projection_last[ps]=*clock;}}
       if(trace3){const int ts=trace3->target_slot[i];if(ts>=0&&trace3->target_first[ts]<0)trace3->target_first[ts]=*clock;}
       if(trace4){const int ss=trace4->source_slot[i],st=trace4->stable_slot[i],ks=trace4->kc_slot[i];if(ss>=0){trace4->source_spikes[ss]++;if(trace4->source_first[ss]<0)trace4->source_first[ss]=*clock;}if(st>=0&&trace4->stable_first[st]<0)trace4->stable_first[st]=*clock;if(ks>=0&&trace4->kc_first[ks]<0){trace4->kc_first[ks]=*clock;if(trace4->kc_order_used<trace4->kc_n){const int q=trace4->kc_order_used++;trace4->kc_order_id[q]=i;trace4->kc_order_step[q]=*clock;}else trace4->overflow=1;}if(i==trace4->target_node&&trace4->target_first[0]<0)trace4->target_first[0]=*clock;}
       if(trace5){const int ss=trace5->source_slot[i];if(ss>=0){trace5->source_spikes[ss]++;if(trace5->source_first[ss]<0)trace5->source_first[ss]=*clock;}if(i==trace5->target_node&&trace5->leader_first[0]<0)trace5->leader_first[0]=*clock;}
       if(trace6){const int ks=trace6->kc_slot[i];if(ks>=0&&trace6->kc_first[ks]<0){trace6->kc_first[ks]=*clock;if(trace6->kc_unique[0]==0)trace6->kc1_step[0]=*clock;if(trace6->kc_unique[0]==9)trace6->kc10_step[0]=*clock;if(trace6->kc_order_used[0]<trace6->kc_n){trace6->kc_order_id[trace6->kc_order_used[0]]=ks;trace6->kc_order_step[trace6->kc_order_used[0]]=*clock;trace6->kc_order_used[0]++;}else trace6->overflow=1;trace6->kc_unique[0]++;}}if(trace7){const int ks=trace7->kc_slot[i];if(ks>=0&&trace7->kc_first[ks]<0){trace7->kc_first[ks]=*clock;if(trace7->kc1_step[0]<0)trace7->kc1_step[0]=*clock;}}if(trace8){const int ks=trace8->kc_slot[i];if(ks>=0&&trace8->kc1_step[0]<0)trace8->kc1_step[0]=*clock;}
       if(kc_mask[i]){
         adaptation[i]+=adaptation_jump;
         eligibility[i]*=std::exp(-dt*(*clock-eligibility_last[i])/tau_elig_ms);
         eligibility[i]+=1.;eligibility_last[i]=*clock;
       }
     }
     // Convex relaxation toward drive+g(t): the bound makes this exact in
     // real arithmetic, not an activity cutoff or a dropped weak connection.
     const float gap=-45.f-rest[i];
     const bool can_fire=v[i]>-45.f || drive[i]>gap || drive[i]+g[i]>gap;
     if(can_fire)active[kept++]=i;else flags[i]=0;
   }
   *nactive=kept;
   for(int q=0;q<queue_count[slot];q++){
     const int i=queue[slot*n+q];
     if(modulation_mask[i]){
       // Keep all outgoing reconstructed edges: deliver transmitter to each
       // actual target. Effects outside the specified memory rule are unknown.
       for(int64_t e=ptr[i];e<ptr[i+1];e++){
         const int j=post[e];
         modulation[j]*=std::exp(-dt*(*clock-modulation_last[j])/100.f);
         modulation[j]+=std::abs(weight[e])/.275f;modulation_last[j]=*clock;
       }
       if(learning_enabled && dan_index[i]>=0){
         for(int p=0;p<nplastic;p++){
           const int pre=plastic_pre[p];
           const double trace=eligibility[pre]*std::exp(-dt*(*clock-eligibility_last[pre])/tau_elig_ms);
           const float gain=dan_gain[dan_index[i]*nplastic+p];
           const int64_t edge=plastic_edge[p];
           const float candidate=weight[edge]*std::exp(-eta*gain*trace);
           const float lower=baseline_weight[p]*floor_fraction;
           weight[edge]=candidate>lower?candidate:lower;
         }
       }
       continue;
     }
     for(int64_t e=ptr[i];e<ptr[i+1];e++){
       const int j=post[e];evolve(j,*clock,drive[j]);
       if(trace6 && refractory[j]==0){const int ks=trace6->kc_slot[j],sc=trace6->source_class[i];if(ks>=0){trace6->all_kc_delivery_count[0]++;if(sc<0||sc>=6){trace6->overflow=1;}else{const int now=*clock;const int ph=trace6->kc_unique[0]==0?0:(trace6->kc_unique[0]<10?1:2);const int z=sc*3+ph;const double w=weight[e];trace6->phase_events[z]++;trace6->phase_signed[z]+=w;if(w>0)trace6->phase_positive[z]+=w;else if(w<0)trace6->phase_negative[z]+=w;if(!trace6->source_seen[i*3+ph]){trace6->source_seen[i*3+ph]=1;trace6->phase_unique_source[z]++;}if(!trace6->target_seen[ks*6*3+sc*3+ph]){trace6->target_seen[ks*6*3+sc*3+ph]=1;trace6->phase_unique_target[z]++;}if(trace6->phase_first[z]<0)trace6->phase_first[z]=now;trace6->phase_last[z]=now;if(trace6->kc_first[ks]<0||now<trace6->kc_first[ks]){const int q=ks*6+sc;trace6->kc_class_events[q]++;trace6->kc_class_signed[q]+=w;if(w>0)trace6->kc_class_positive[q]+=w;else if(w<0)trace6->kc_class_negative[q]+=w;}}}}
       if(refractory[j]==0){const float delivered=intervene_s2n(i,j,e);g[j]+=delivered;observe_s2m(i,j,e);if(trace5){const int ss=trace5->source_slot[i];if(ss>=0 && j==trace5->target_node){const int st=*clock;trace5->source_events[ss]++;trace5->source_signed[ss]+=weight[e];if(weight[e]>0)trace5->source_positive[ss]+=weight[e];else if(weight[e]<0)trace5->source_negative[ss]+=weight[e];if(trace5->source_first_delivery[ss]<0)trace5->source_first_delivery[ss]=st;trace5->source_last_delivery[ss]=st;if(trace5->leader_first[0]<0 || st<trace5->leader_first[0]){trace5->source_pre_count[ss]++;if(trace5->source_pre_first[ss]<0)trace5->source_pre_first[ss]=st;trace5->source_pre_last[ss]=st;}}}if(trace4){const int ss=trace4->source_slot[i];if(ss>=0 && j==trace4->target_node){const int st=*clock;trace4->source_events[ss]++;trace4->source_signed[ss]+=weight[e];if(weight[e]>0)trace4->source_positive[ss]+=weight[e];else if(weight[e]<0)trace4->source_negative[ss]+=weight[e];if(trace4->source_first_delivery[ss]<0)trace4->source_first_delivery[ss]=st;trace4->source_last_delivery[ss]=st;if(trace4->target_first[0]<0 || st<trace4->target_first[0]){trace4->source_pre_count[ss]++;if(trace4->source_pre_first[ss]<0)trace4->source_pre_first[ss]=st;trace4->source_pre_last[ss]=st;}}}if(trace){const int64_t ts=*clock-trace->start_clock;const int cs=trace->candidate_slot[i];const bool ctarget=trace->cb_mask[j];const int ks=trace->kc_slot[j];if(ts<0||ts>=trace->steps)trace->overflow=1;else if(cs>=0&&ctarget){const int z=cs*trace->steps+ts;trace->candidate_event_counts[z]++;const double w=weight[e];trace->candidate_signed[z]+=w;if(w>0)trace->candidate_positive[z]+=w;else if(w<0)trace->candidate_negative[z]+=w;}else if(trace->cb_mask[i]&&trace->kc_slot[i]<0&&ks>=0){trace->cbkc_event_counts[ts]++;const double w=weight[e];trace->cbkc_signed[ts]+=w;if(w>0)trace->cbkc_positive[ts]+=w;else if(w<0)trace->cbkc_negative[ts]+=w;}}if(trace2){const int ps=trace2->projection_slot[i],source_cb_slot=trace2->cb_slot[i],target_cb_slot=trace2->cb_slot[j],ks=trace2->kc_slot[j];int phase=trace2->kc_unique<10?0:trace2->kc_unique<100?1:trace2->kc_unique<1000?2:3;if(ps>=0&&target_cb_slot>=0){int z=ps*4+phase;double w=weight[e];if(trace2->projection_first_delivery[ps]<0)trace2->projection_first_delivery[ps]=*clock;trace2->projection_events[z]++;trace2->projection_signed[z]+=w;if(w>0)trace2->projection_positive[z]+=w;else if(w<0)trace2->projection_negative[z]+=w;}else if(source_cb_slot>=0&&ks>=0){int z=source_cb_slot*4+phase;double w=weight[e];if(trace2->cb_first[source_cb_slot]<0)trace2->cb_first[source_cb_slot]=*clock;trace2->cb_events[z]++;trace2->cb_signed[z]+=w;if(w>0)trace2->cb_positive[z]+=w;else if(w<0)trace2->cb_negative[z]+=w;}}if(trace3){const int ss=trace3->source_slot[i],tt=trace3->target_slot[j];if(ss>=0&&tt>=0&&trace3->target_first[tt]<0){const int z=ss*trace3->target_n+tt;const double w=weight[e];trace3->pair_events[z]++;trace3->pair_signed[z]+=w;if(w>0)trace3->pair_positive[z]+=w;else if(w<0)trace3->pair_negative[z]+=w;if(trace3->pair_first[z]<0)trace3->pair_first[z]=*clock;trace3->pair_last[z]=*clock;}}if(trace4&&trace4->out_used<trace4->out_capacity&&i==trace4->target_node){const int64_t u=trace4->out_used++;trace4->out_step[u]=*clock;trace4->out_target[u]=j;trace4->out_weight[u]=weight[e];}else if(trace4&&i==trace4->target_node){trace4->overflow=1;}if(trace4){const int ss=trace4->source_slot[i];if(ss>=0&&j==trace4->target_node){const int z=ss;const double w=weight[e];trace4->source_events[z]++;trace4->source_signed[z]+=w;if(w>0)trace4->source_positive[z]+=w;else if(w<0)trace4->source_negative[z]+=w;}}awaken(j);}
     }
   }
   queue_count[slot]=0;
   for(int q=0;q<queue_count[future];q++){
     const int i=queue[future*n+q];v[i]=rest[i];g[i]=0.f;refractory[i]=rfc;
   }
 }
 // Materialize all states at the observation boundary (no threshold can be
 // missed in sleeping cells). This also supports auditable voltage readouts.
 for(int i=0;i<n;i++)evolve(i,*clock-1,drive[i]);
}
