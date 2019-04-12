REPS = 100;   minTime = Inf;   
tic;  % TIC, pair 1

for i=1:REPS
   tStart = tic;  % TIC, pair 2     
   mdyn_inv = inv(mdyn);
   tElapsed = toc(tStart);  % TOC, pair 2  
   minTime = min(tElapsed, minTime);
end
averageTime = toc/REPS;  % TOC, pair 1  