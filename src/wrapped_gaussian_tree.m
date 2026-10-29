function [logDensity, detail] = wrapped_gaussian_tree(residual, covariance, periods, tolerance, maxExpansions)
periods = periods(:);
setupClock = tic;
dimension = numel(periods);
assert(size(residual,1) == dimension && isequal(size(covariance),[dimension dimension]));
assert(all(periods > 0) && tolerance > 0 && tolerance < 1 && maxExpansions >= dimension);
scaledCovariance = covariance ./ (periods * periods');
root = chol(scaledCovariance,'lower');
conditionalSigma = diag(root);
logPeak = zeros(dimension,1);
for j = 1:dimension
    [lower, remainder] = wrapped_normal_interval(0,conditionalSigma(j)^2,1,1e-13);
    logPeak(j) = add_logs(lower,remainder);
end
remainingPeak = [flipud(cumsum(flipud(logPeak)));0];
scaledResidual = mod(residual ./ periods + .5,1) - .5;
pointCount = size(residual,2);
logDensity = zeros(1,pointCount);
logError = zeros(1,pointCount);
logMaximum = zeros(1,pointCount);
expansions = zeros(1,pointCount);
leaves = zeros(1,pointCount);
peakFrontier = zeros(1,pointCount);
pointSeconds = zeros(1,pointCount);
logJacobian = sum(log(periods));
setupSeconds = toc(setupClock);
for point = 1:pointCount
    pointClock = tic;
    t = scaledResidual(:,point);
    capacity = 256;
    innovations = zeros(dimension,capacity);
    depth = ones(1,capacity);
    radius = -ones(1,capacity);
    logPrefix = zeros(1,capacity);
    logBound = -inf(1,capacity);
    logBound(1) = remainingPeak(1);
    count = 1;
    total = -inf;
    maximum = -inf;
    remainder = logBound(1);
    used = 0;
    leafCount = 0;
    largestFrontier = 1;
    while remainder > total + log(tolerance) && used < maxExpansions
        [~, chosen] = max(logBound(1:count));
        j = depth(chosen);
        previous = innovations(:,chosen);
        prefix = logPrefix(chosen);
        nextRadius = radius(chosen) + 1;
        center = root(j,1:j-1) * previous(1:j-1) - t(j);
        nearest = round(center);
        displacement = nearest - center;
        if nextRadius == 0
            offsets = 0;
        else
            offsets = [-nextRadius,nextRadius];
        end
        if used + numel(offsets) > maxExpansions
            break
        end
        radius(chosen) = nextRadius;
        logBound(chosen) = prefix + scalar_tail(displacement,nextRadius,conditionalSigma(j)) + remainingPeak(j+1);
        for offset = offsets
            standardized = (displacement + offset) / conditionalSigma(j);
            value = prefix - .5*standardized^2 - log(conditionalSigma(j)) - .5*log(2*pi);
            if j == dimension
                total = add_logs(total,value);
                maximum = max(maximum,value);
                leafCount = leafCount + 1;
            else
                count = count + 1;
                if count > capacity
                    capacity = capacity + 256;
                    innovations(:,capacity) = 0;
                    depth(capacity) = 1;
                    radius(capacity) = -1;
                    logPrefix(capacity) = 0;
                    logBound(capacity) = -inf;
                end
                innovations(:,count) = previous;
                innovations(j,count) = standardized;
                depth(count) = j+1;
                radius(count) = -1;
                logPrefix(count) = value;
                logBound(count) = value + remainingPeak(j+1);
            end
        end
        used = used + numel(offsets);
        largestFrontier = max(largestFrontier,count);
        top = max(logBound(1:count));
        remainder = top + log(sum(exp(logBound(1:count)-top)));
    end
    logDensity(point) = total-logJacobian;
    logError(point) = remainder-logJacobian;
    logMaximum(point) = maximum-logJacobian;
    expansions(point) = used;
    leaves(point) = leafCount;
    peakFrontier(point) = largestFrontier;
    pointSeconds(point) = toc(pointClock);
end
detail = struct('logErrorBound',logError,'relativeErrorBound',exp(logError-logDensity), ...
    'converged',logError-logDensity <= log(tolerance), 'logMaximumComponent',logMaximum, ...
    'expansions',expansions,'leaves',leaves,'peakFrontier',peakFrontier, ...
    'pointSeconds',pointSeconds,'setupSeconds',setupSeconds, ...
    'logGlobalUpper',remainingPeak(1)-logJacobian, ...
    'logRemainingUpper',remainingPeak(2)-sum(log(periods(2:end))));
end

function value = scalar_tail(displacement,radius,sigma)
distances = radius+1+[displacement,-displacement];
argument = distances/(sqrt(2)*sigma);
logDensity = -argument.^2-log(sqrt(2*pi)*sigma);
logIntegral = log(.5)+log(erfcx(argument))-argument.^2;
left = add_logs(logDensity(1),logIntegral(1));
right = add_logs(logDensity(2),logIntegral(2));
value = add_logs(left,right);
end

function value = add_logs(a,b)
if isinf(a) && a < 0
    value = b;
elseif isinf(b) && b < 0
    value = a;
else
    top = max(a,b);
    value = top+log1p(exp(min(a,b)-top));
end
end
