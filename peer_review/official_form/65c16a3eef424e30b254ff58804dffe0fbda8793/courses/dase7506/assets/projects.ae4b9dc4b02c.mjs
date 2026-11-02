// MP1 is displayed from its immutable archive; automatic sync accepts MP2.
export const PROTOCOL = '7506-submissions-v1';
export const PROJECTS = {
  mp1: {name:'Small Language Model Challenge', unit:'BPB', lower:true, open:true, min:0, max:null, review_threshold:null, review_request_flag_above:3, reward:0, penalty:0, score_deadline:'2026-10-01T00:00:00+08:00', review_days:7},
  mp2: {name:'Selecting Reliable LLM Reasoning', unit:'%', lower:false, open:true, min:0, max:100, review_threshold:null, review_request_flag_above:3, reward:0, penalty:0, score_deadline:'2026-11-01T00:00:00+08:00', review_days:7},
  mp3: {name:'Mini-project 3', unit:'Score', lower:false, open:false, min:null, max:null, review_threshold:null, reward:0, penalty:0},
};
