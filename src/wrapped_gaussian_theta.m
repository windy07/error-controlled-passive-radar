function [logDensity,detail] = wrapped_gaussian_theta(residual,covariance,periods,tolerance,maxExpansions,prepared)
setupClock = tic;
periods = periods(:);
dimension = numel(periods);
assert(size(residual,1)==dimension && all(periods>0) && tolerance>0 && tolerance<1);
normalizedCovariance = covariance./(periods*periods');
root = chol(normalizedCovariance,'lower');
logNormalizer = -sum(log(diag(root)))-dimension/2*log(2*pi);
logJacobian = sum(log(periods));
if nargin<6
    setup = theta_lattice_setup(normalizedCovariance);
else
    setup = prepared;
end
t = mod(residual./periods+.5,1)-.5;
pointCount = size(t,2);
initialOmega = 1i*(normalizedCovariance\eye(dimension))/(2*pi);
argument = initialOmega*t;
phaseAngle = zeros(1,pointCount);
logScale = logNormalizer-logJacobian;
for index=1:numel(setup.operations)
    operation = setup.operations{index};
    if operation.kind=="basis"
        argument = operation.matrix.'*argument;
    elseif operation.kind=="shift"
        argument = argument+operation.vector;
    else
        first = argument(1,:);
        phaseAngle = phaseAngle-.5*angle(-1i*operation.pivot)-pi*real(first.^2/operation.pivot);
        logScale = logScale-.5*log(abs(operation.pivot));
        argument = [first/operation.pivot;argument(2:end,:)-operation.column*first/operation.pivot];
    end
end
omega = setup.omega;
imaginary = imag(omega);
shift = round(imaginary\imag(argument));
argument = argument-omega*shift;
argument = argument-round(real(argument));
phaseAngle = phaseAngle-pi*real(sum(shift.*(omega*shift),1))-2*pi*real(sum(shift.*argument,1));
centres = -(imaginary\imag(argument));
logScales = repmat(logScale,1,pointCount);
phases = exp(1i*phaseAngle);
upper = sqrt(pi)*chol(imaginary);
logPeak = zeros(dimension,1);
for index=1:dimension
    [lower,error] = wrapped_normal_interval(0,root(index,index)^2,1,1e-13);
    logPeak(index) = add_logs(lower,error);
end
logGlobalUpper = sum(logPeak)-logJacobian;
setupSeconds = toc(setupClock);
logDensity = zeros(1,pointCount);logError = zeros(1,pointCount);
pointSeconds = zeros(1,pointCount);expansions = zeros(1,pointCount);
leaves = zeros(1,pointCount);radii = zeros(1,pointCount);hitBudget = false(1,pointCount);
imaginaryRemainder = zeros(1,pointCount);
numericalFailure = false(1,pointCount);
for point=1:pointCount
    pointClock = tic;
    innovation = zeros(dimension,1);
    for index=1:dimension
        babaiCentre = root(index,1:index-1)*innovation(1:index-1,1)-t(index,point);
        innovation(index) = (round(babaiCentre)-babaiCentre)/root(index,index);
    end
    logBabai = logNormalizer-.5*sum(innovation.^2)-logJacobian;
    logTarget = log(tolerance/2)+logBabai-logScales(point);
    low = sqrt(dimension/2)+setup.rho/2;
    high = max(1,low);
    while log_tail(high,dimension,setup.rho)>logTarget
        high = 2*high;
    end
    for iteration=1:45
        middle = (low+high)/2;
        if log_tail(middle,dimension,setup.rho)>logTarget
            low = middle;
        else
            high = middle;
        end
    end
    radiusSquared = high^2;
    candidate = zeros(dimension,1);
    used = 0;leafCount = 0;exceeded = false;
    top = -inf;oscillatory = 0;
    latticeCentre = centres(:,point);
    visit(dimension,0);
    if exceeded
        logDensity(point) = logBabai;
        logError(point) = logGlobalUpper;
    else
        rotated = phases(point)*oscillatory;
        logAbsoluteError = logScales(point)+log_tail(high,dimension,setup.rho);
        logApproximation = logScales(point)+top+log(max(real(rotated),realmin));
        if real(rotated)<=0 || logAbsoluteError>=logApproximation
            numericalFailure(point) = true;
            logDensity(point) = logBabai;
            logError(point) = logGlobalUpper;
        else
            lower = logApproximation+log1p(-exp(logAbsoluteError-logApproximation));
            logDensity(point) = max(logBabai,lower);
            logError(point) = log(2)+logAbsoluteError;
            imaginaryRemainder(point) = abs(imag(rotated))/real(rotated);
        end
    end
    pointSeconds(point) = toc(pointClock);
    expansions(point) = used;leaves(point) = leafCount;radii(point) = high;hitBudget(point) = exceeded;
end
detail = struct('logErrorBound',logError,'relativeErrorBound',exp(logError-logDensity), ...
    'converged',~hitBudget & ~numericalFailure & logError-logDensity<=log(tolerance)+1e-10, ...
    'expansions',expansions,'leaves',leaves,'radius',radii,'hitBudget',hitBudget, ...
    'pointSeconds',pointSeconds,'setupSeconds',setupSeconds,'reduction',setup, ...
    'imaginaryRemainder',imaginaryRemainder,'numericalFailure',numericalFailure,'logGlobalUpper',logGlobalUpper, ...
    'logRemainingUpper',sum(logPeak(2:end))-sum(log(periods(2:end))));

    function visit(level,distance)
        centre = latticeCentre(level)-upper(level,level+1:end)* ...
            (candidate(level+1:dimension,1)-latticeCentre(level+1:dimension,1))/upper(level,level);
        width = sqrt(max(0,radiusSquared-distance))/abs(upper(level,level));
        integers = ceil(centre-width):floor(centre+width);
        [~,order] = sort(abs(integers-centre));
        for integer=integers(order)
            if used>=maxExpansions,exceeded=true;return;end
            used = used+1;
            candidate(level) = integer;
            next = distance+(upper(level,level)*(integer-centre))^2;
            if level==1
                termPhase = pi*candidate.'*real(omega)*candidate+2*pi*candidate.'*real(argument(:,point));
                term = exp(1i*termPhase);
                if -next>top
                    oscillatory = oscillatory*exp(top+next)+term;
                    top = -next;
                else
                    oscillatory = oscillatory+exp(-next-top)*term;
                end
                leafCount = leafCount+1;
            else
                visit(level-1,next);
                if exceeded,return;end
            end
        end
    end
end

function value = log_tail(radius,dimension,rho)
argument = (radius-rho/2)^2;
shape = dimension/2;
if mod(dimension,2)==0
    currentShape = 1;
    gammaUpper = -argument;
else
    currentShape = .5;
    gammaUpper = .5*log(pi)+log(erfcx(sqrt(argument)))-argument;
end
while currentShape<shape
    gammaUpper = add_logs(log(currentShape)+gammaUpper,currentShape*log(argument)-argument);
    currentShape = currentShape+1;
end
value = log(dimension/2)+dimension*log(2/rho)+gammaUpper;
end

function value = add_logs(left,right)
if isinf(left) && left<0,value=right;return;end
if isinf(right) && right<0,value=left;return;end
top = max(left,right);
value = top+log1p(exp(min(left,right)-top));
end
