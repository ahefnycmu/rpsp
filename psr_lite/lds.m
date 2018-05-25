function out = lds(obs, act, test_obs, test_act, future_win, past_win, options)

if nargin < 7 || isempty(options); options = struct; end;
if ~isfield(options, 'nonblind'); options.nonblind = 0; end;
if ~isfield(options, 'p'); options.p = 'best'; end;
if ~isfield(options, 'choose_p'); options.choose_p = 0; end;

fut = future_win;
past = past_win;

num_series = length(obs);
data = iddata(obs{1}, act{1}, 1);

for i = 2:num_series
    data = merge(data, iddata(obs{i}, act{i}, 1));
end

alg = 'auto';
if options.nonblind; alg = 'SSARX'; end;
opt = n4sidOptions('N4Weight', alg, 'N4Horizon', [fut past past]);
if(options.choose_p); p = 1:options.p; else p = options.p; end;
lds = n4sid(data, p, opt);

horizon = future_win;
M = length(test_obs);
est_obs = cell(M,1);

for i = 1:M    
    data = iddata(test_obs{i}, test_act{i}, 1);
    
    [N,d] = size(test_obs{i});    
    est_obs{i} = zeros(horizon,N,d);
    
    for k=1:horizon
        
        oh = predict(lds, data, k);
        est_obs{i}(k,:,:) = oh.OutputData;
    end
end

out = est_obs;
