function result = finite_grid_posterior(coarse, fine, prediction, points, models, method, options)
[sources,nodes] = size(coarse);
pointCount = size(points,2);
logPrior = -log(pointCount)*ones(1,pointCount);
if isfield(options,'logPrior')
    assert(isequal(size(options.logPrior),[1,pointCount]) && all(~isnan(options.logPrior)));
    logPrior = options.logPrior-log_sum(options.logPrior);
    assert(all(isfinite(logPrior) | logPrior==-inf) && any(isfinite(logPrior)));
end
maxQueries = pointCount;
if isfield(options,'maxQueries'),maxQueries = min(pointCount,options.maxQueries);end
assert(maxQueries>=1 && maxQueries==floor(maxQueries) && (options.prune || maxQueries==pointCount));
assert(isequal(size(fine),size(coarse)) && size(prediction,1) == sources*nodes && size(prediction,2) == pointCount);
assert(numel(models) == nodes && options.posteriorTolerance >= 0 && options.posteriorTolerance < 1);
started = tic;
progressEvery = inf;
if isfield(options,'progressEvery'),progressEvery = options.progressEvery;end
nextProgress = progressEvery;
progressSeconds = inf;
if isfield(options,'progressSeconds'),progressSeconds = options.progressSeconds;end
nextProgressTime = progressSeconds;
usePairEnvelope = false;
if isfield(options,'pairEnvelope'),usePairEnvelope = options.pairEnvelope;end
assert(isscalar(usePairEnvelope) && (~usePairEnvelope || sources>=2));
useMajorantEnvelope = false;
if isfield(options,'majorantEnvelope'),useMajorantEnvelope = options.majorantEnvelope;end
assert(isscalar(useMajorantEnvelope));
pairAfterQueries = 0;
if isfield(options,'pairAfterQueries'),pairAfterQueries = options.pairAfterQueries;end
assert(isscalar(pairAfterQueries) && isfinite(pairAfterQueries) && ...
    pairAfterQueries>=0 && pairAfterQueries==floor(pairAfterQueries));
pairPending = usePairEnvelope && options.prune && options.envelope=="prefix" && ...
    method~="coarse" && pairAfterQueries>0;
nodePrefixes = cell(1,nodes);
envelopeRefinements = 0;
deferredEnvelopeSeconds = 0;
factorOrder = "input";
if isfield(options,'factorOrder'),factorOrder = options.factorOrder;end
assert(isscalar(factorOrder) && ismember(factorOrder,["input","variance"]));
if method == "maximum"
    factorEvaluator = @wrapped_gaussian_maximum;
elseif options.backend == "tree"
    factorEvaluator = @wrapped_gaussian_tree;
elseif options.backend == "sphere"
    factorEvaluator = @wrapped_gaussian_sphere;
elseif options.backend == "sphere_batch"
    factorEvaluator = @wrapped_gaussian_sphere_batch;
elseif options.backend == "theta"
    factorEvaluator = @wrapped_gaussian_theta;
elseif options.backend == "box"
    factorEvaluator = @(r,C,L,tol,budget) wrapped_gaussian_box(r,C,L,tol,budget,options.boxRadius);
else
    error('FiniteGrid:Backend','Unknown lattice evaluation algorithm.');
end
coarseLog = zeros(1,pointCount);
fineResidual = cell(1,nodes);
conditionalCovariance = cell(1,nodes);
factorPeriods = cell(1,nodes);
factorPeak = zeros(1,nodes);
prefixEnvelope = zeros(1,pointCount);
extraArguments = repmat({{}},1,nodes);
thetaCovariances = cell(1,nodes);
for n = 1:nodes
    model = models{n};
    predicted = prediction((n-1)*sources+(1:sources),:);
    if method == "difference"
        residual = coarse(:,n)-predicted;
        calibrated = residual-model.basis*(model.flatCalibrationGain*residual);
        normalized = model.noiseCoarseRoot \ calibrated;
        coarseLog = coarseLog-.5*sum(normalized.^2,1);
        fineResidual{n} = fine(:,n)-predicted-model.flatConditionalGain*residual;
        conditionalCovariance{n} = model.flatConditionalCovariance;
    else
        residual = coarse(:,n)-predicted-model.meanBias;
        normalized = model.coarseRoot \ residual;
        coarseLog = coarseLog-.5*sum(normalized.^2,1);
        fineResidual{n} = fine(:,n)-predicted-model.meanBias-model.conditionalGain*residual;
        conditionalCovariance{n} = model.conditionalCovariance;
    end
    periods = model.periods;
    if factorOrder == "variance" && method ~= "coarse"
        [~,permutation] = sort(diag(conditionalCovariance{n})./periods.^2);
        fineResidual{n} = fineResidual{n}(permutation,:);
        conditionalCovariance{n} = conditionalCovariance{n}(permutation,permutation);
        periods = periods(permutation);
    end
    factorPeriods{n} = periods;
    if options.backend == "theta" && ~ismember(method,["coarse","maximum"])
        thetaCovariances{n} = conditionalCovariance{n}./(periods*periods');
        previousMatch = 0;
        for previous = 1:n-1
            if isequal(thetaCovariances{n},thetaCovariances{previous})
                previousMatch = previous;
                break
            end
        end
        if previousMatch == 0
            extraArguments{n} = {theta_lattice_setup(thetaCovariances{n})};
        else
            extraArguments{n} = extraArguments{previousMatch};
        end
    end
    if method == "maximum"
        root = chol(conditionalCovariance{n},'lower');
        factorPeak(n) = -sum(log(diag(root)))-sources/2*log(2*pi);
        if options.envelope == "prefix"
            firstResidual = mod(fineResidual{n}(1,:)+periods(1)/2,periods(1))-periods(1)/2;
            nodeEnvelope = factorPeak(n)-firstResidual.^2/(2*conditionalCovariance{n}(1,1));
        end
    elseif method ~= "coarse"
        [peak,detail] = factorEvaluator(zeros(sources,1),conditionalCovariance{n},periods,1e-8,options.maxExpansions,extraArguments{n}{:});
        factorPeak(n) = min(detail.logGlobalUpper,add_logs(peak,detail.logErrorBound));
        if options.envelope == "prefix"
            [first,firstError] = wrapped_normal_interval(fineResidual{n}(1,:),conditionalCovariance{n}(1,1),periods(1),1e-10);
            nodeEnvelope = min(factorPeak(n),add_logs(first,firstError)+detail.logRemainingUpper);
        end
    end
    if options.envelope == "prefix" && method ~= "coarse"
        if useMajorantEnvelope
            nodeEnvelope = min(nodeEnvelope,wrapped_gaussian_majorant(fineResidual{n}, ...
                conditionalCovariance{n},periods,method));
        end
        if pairPending
            nodePrefixes{n} = nodeEnvelope;
        elseif usePairEnvelope
            nodeEnvelope = min(nodeEnvelope,pair_envelope(fineResidual{n}, ...
                conditionalCovariance{n},periods,method));
        end
        prefixEnvelope = prefixEnvelope+nodeEnvelope;
    end
end
preparationSeconds = toc(started);
if method == "coarse"
    logWeights = coarseLog+logPrior;
    posterior = exp(logWeights-max(logWeights));
    posterior = posterior/sum(posterior);
    [~,best] = max(logWeights);
    posteriorMean = points*posterior';
    centered = points-posteriorMean;
    result = struct('mean',posteriorMean,'covariance',(centered.*posterior)*centered','map',points(:,best),'posterior',posterior, ...
        'tvBound',0,'converged',true,'queried',pointCount,'queryFraction',1,'seconds',toc(started), ...
        'latticeExpansions',0,'latticeLeaves',0,'unconvergedFactors',0,'mapLogGapBound',0,'noPositiveMass',false, ...
        'preparationSeconds',preparationSeconds,'rankingSeconds',0,'evaluationSeconds',0, ...
        'envelopeRefinements',0,'deferredEnvelopeSeconds',0, ...
        'logLowerMass',log_sum(logWeights),'logRemainderMass',-inf);
    return
end
if options.envelope == "prefix"
    logEnvelope = coarseLog+prefixEnvelope+logPrior;
else
    logEnvelope = coarseLog+sum(factorPeak)+logPrior;
end
if options.prune
    [sortedEnvelope,order] = sort(logEnvelope,'descend');
    tail = -inf(1,pointCount+1);
    for i = pointCount:-1:1
        tail(i) = add_logs(sortedEnvelope(i),tail(i+1));
    end
else
    order = 1:pointCount;
    tail = -inf(1,pointCount+1);
end
rankingSeconds = toc(started)-preparationSeconds;
evaluationClock = tic;
logLower = -inf(1,pointCount);
logUpper = -inf(1,pointCount);
lowerMass = -inf;
queriedUncertainty = -inf;
evaluated = 0;
totalExpansions = 0;
totalLeaves = 0;
unconvergedFactors = 0;
tvBound = 1;
while evaluated < maxQueries
    chosen = order(evaluated+1:min(evaluated+options.batchSize,maxQueries));
    lower = coarseLog(chosen)+logPrior(chosen);
    upper = lower;
    relativeLogGap = zeros(size(lower));
    for n = 1:nodes
        [factor,detail] = factorEvaluator(fineResidual{n}(:,chosen),conditionalCovariance{n}, ...
            factorPeriods{n},options.factorTolerance,options.maxExpansions,extraArguments{n}{:});
        if method == "maximum"
            factor = detail.logMaximumComponent;
            detail.converged = detail.logErrorBound-factor<=log(options.factorTolerance);
        end
        lower = lower+factor;
        upper = upper+add_logs(factor,detail.logErrorBound);
        relativeError = detail.logErrorBound-factor;
        relativeLogGap = relativeLogGap+max(relativeError,0)+log1p(exp(-abs(relativeError)));
        totalExpansions = totalExpansions+sum(detail.expansions);
        totalLeaves = totalLeaves+sum(detail.leaves);
        unconvergedFactors = unconvergedFactors+sum(~detail.converged);
    end
    logLower(chosen) = lower;
    logUpper(chosen) = upper;
    lowerMass = add_logs(lowerMass,log_sum(lower));
    logRelativeDifference = relativeLogGap+log(-expm1(-relativeLogGap));
    logDifference = lower+logRelativeDifference;
    zeroLower = ~isfinite(lower);
    logDifference(zeroLower) = upper(zeroLower);
    queriedUncertainty = add_logs(queriedUncertainty,log_sum(logDifference));
    evaluated = evaluated+numel(chosen);
    if options.prune
        unqueried = tail(evaluated+1);
    elseif evaluated < pointCount
        tvBound = 1;
        continue
    else
        unqueried = -inf;
    end
    remaining = add_logs(queriedUncertainty,unqueried);
    tvBound = exp(remaining-add_logs(lowerMass,remaining));
    if pairPending && evaluated>=pairAfterQueries && evaluated<pointCount && ...
            tvBound>options.posteriorTolerance
        refinementClock = tic;
        remainingIds = order(evaluated+1:end);
        refinedPrefix = zeros(1,numel(remainingIds));
        for n = 1:nodes
            correlatedUpper = pair_envelope(fineResidual{n}(:,remainingIds), ...
                conditionalCovariance{n},factorPeriods{n},method);
            refinedPrefix = refinedPrefix+min(nodePrefixes{n}(remainingIds),correlatedUpper);
        end
        refinedEnvelope = coarseLog(remainingIds)+logPrior(remainingIds)+refinedPrefix;
        addedPreparation = toc(refinementClock);
        preparationSeconds = preparationSeconds+addedPreparation;
        refinementRankingClock = tic;
        [remainingSorted,permutation] = sort(refinedEnvelope,'descend');
        order(evaluated+1:end) = remainingIds(permutation);
        sortedEnvelope(evaluated+1:end) = remainingSorted;
        for i = pointCount:-1:evaluated+1
            tail(i) = add_logs(sortedEnvelope(i),tail(i+1));
        end
        addedRanking = toc(refinementRankingClock);
        rankingSeconds = rankingSeconds+addedRanking;
        deferredEnvelopeSeconds = addedPreparation+addedRanking;
        envelopeRefinements = 1;
        pairPending = false;
        nodePrefixes = {};
        remaining = add_logs(queriedUncertainty,tail(evaluated+1));
        tvBound = exp(remaining-add_logs(lowerMass,remaining));
    end
    if evaluated>=nextProgress || toc(started)>=nextProgressTime
        fprintf('    Grid %d/%d: TV %.3g, unresolved factors %d, %.1f s.\n', ...
            evaluated,pointCount,tvBound,unconvergedFactors,toc(started));
        nextProgress = evaluated+progressEvery;
        nextProgressTime = toc(started)+progressSeconds;
    end
    if options.prune && tvBound <= options.posteriorTolerance
        break
    end
end
evaluationSeconds = toc(evaluationClock)-deferredEnvelopeSeconds;
if ~isfinite(lowerMass)
    result = struct('mean',nan(2,1),'covariance',nan(2),'map',nan(2,1),'posterior',nan(1,pointCount), ...
        'tvBound',1,'converged',false,'queried',evaluated,'queryFraction',evaluated/pointCount, ...
        'seconds',toc(started),'latticeExpansions',totalExpansions,'latticeLeaves',totalLeaves, ...
        'unconvergedFactors',unconvergedFactors,'mapLogGapBound',inf,'noPositiveMass',true, ...
        'preparationSeconds',preparationSeconds,'rankingSeconds',rankingSeconds,'evaluationSeconds',evaluationSeconds, ...
        'envelopeRefinements',envelopeRefinements,'deferredEnvelopeSeconds',deferredEnvelopeSeconds, ...
        'logLowerMass',-inf,'logRemainderMass',remaining);
    return
end
posterior = exp(logLower-max(logLower));
posterior = posterior/sum(posterior);
[bestValue,best] = max(logLower);
maximumUpper = max(logUpper);
if evaluated < pointCount
    maximumUpper = max(maximumUpper,sortedEnvelope(evaluated+1));
end
posteriorMean = points*posterior';
centered = points-posteriorMean;
result = struct('mean',posteriorMean,'covariance',(centered.*posterior)*centered','map',points(:,best),'posterior',posterior, ...
    'tvBound',tvBound,'converged',tvBound <= options.posteriorTolerance, ...
    'queried',evaluated,'queryFraction',evaluated/pointCount,'seconds',toc(started), ...
    'latticeExpansions',totalExpansions,'latticeLeaves',totalLeaves,'unconvergedFactors',unconvergedFactors, ...
    'mapLogGapBound',maximumUpper-bestValue,'noPositiveMass',false, ...
    'preparationSeconds',preparationSeconds,'rankingSeconds',rankingSeconds,'evaluationSeconds',evaluationSeconds, ...
    'envelopeRefinements',envelopeRefinements,'deferredEnvelopeSeconds',deferredEnvelopeSeconds, ...
    'logLowerMass',log_sum(logLower),'logRemainderMass',remaining);
end

function upper = pair_envelope(residual,covariance,periods,method)
[~,sourceOrder] = sort(diag(covariance)./periods.^2);
pairIndices = sourceOrder(1:2);
[pair,detail] = wrapped_gaussian_box(residual(pairIndices,:), ...
    covariance(pairIndices,pairIndices),periods(pairIndices),1e-10,49,3);
conditionalRoot = chol(covariance(sourceOrder,sourceOrder),'lower');
suffixPeak = 0;
for coordinate = 3:numel(periods)
    if method == "maximum"
        suffixPeak = suffixPeak-log(conditionalRoot(coordinate,coordinate))-.5*log(2*pi);
    else
        [peakLower,peakError] = wrapped_normal_interval(0, ...
            conditionalRoot(coordinate,coordinate)^2,periods(sourceOrder(coordinate)),1e-13);
        suffixPeak = suffixPeak+add_logs(peakLower,peakError);
    end
end
upper = add_logs(pair,detail.logErrorBound)+suffixPeak;
end

function value = log_sum(values)
top = max(values);
if isinf(top) && top < 0
    value = -inf;
else
    value = top+log(sum(exp(values-top)));
end
end

function value = add_logs(a,b)
top = max(a,b);
value = top+log1p(exp(min(a,b)-top));
bothZero = isinf(top) & top < 0;
value(bothZero) = -inf;
end
