function [logDensity,detail] = wrapped_gaussian_sphere_batch(residual,covariance,periods,tolerance,maxExpansions)
periods=periods(:);dimension=numel(periods);
started=tic;
assert(size(residual,1)==dimension && all(periods>0) && tolerance>0 && tolerance<1);
root=chol(covariance./(periods*periods'),'lower');
sigma=diag(root);logPeak=zeros(dimension,1);
for j=1:dimension
    [lower,errorBound]=wrapped_normal_interval(0,sigma(j)^2,1,1e-13);
    logPeak(j)=add_logs(lower,errorBound);
end
logNormalizer=-sum(log(sigma))-dimension/2*log(2*pi);
logGlobalPeak=sum(logPeak);logJacobian=sum(log(periods));
tAll=mod(residual./periods+.5,1)-.5;
pointCount=size(residual,2);
innovations=zeros(dimension,pointCount);
centers=zeros(dimension,pointCount);
for j=1:dimension
    centers(j,:)=root(j,1:j-1)*innovations(1:j-1,:)-tAll(j,:);
    innovations(j,:)=(round(centers(j,:))-centers(j,:))/sigma(j);
end
babaiDistance=sum(innovations.^2,1);
logBabai=logNormalizer-.5*babaiDistance;
target=log(tolerance)+logBabai-logGlobalPeak;
low=dimension*ones(1,pointCount);
high=max(dimension+1,babaiDistance+1);
needsExpansion=.5*(dimension*(1+log(high/dimension))-high)>target;
while any(needsExpansion)
    high(needsExpansion)=2*high(needsExpansion);
    needsExpansion=.5*(dimension*(1+log(high/dimension))-high)>target;
end
for iteration=1:45
    middle=(low+high)/2;
    moveLow=.5*(dimension*(1+log(middle/dimension))-middle)>target;
    low(moveLow)=middle(moveLow);
    high(~moveLow)=middle(~moveLow);
end
radiiSquared=high;
singleLeaf=repmat(maxExpansions>=dimension,1,pointCount);
distance=zeros(1,pointCount);
for j=1:dimension
    width=sigma(j)*sqrt(max(0,radiiSquared-distance));
    first=ceil(centers(j,:)-width);last=floor(centers(j,:)+width);
    singleLeaf=singleLeaf & first==last & first==round(centers(j,:));
    distance=distance+innovations(j,:).^2;
end
logDensity=logBabai-logJacobian;
logError=logGlobalPeak+.5*(dimension*(1+log(radiiSquared/dimension))-radiiSquared)-logJacobian;
logMaximum=logDensity;
expansionCount=dimension*ones(1,pointCount);leafCount=ones(1,pointCount);
hitBudget=false(1,pointCount);
pointSeconds=nan(1,pointCount);
setupSeconds=toc(started);
enumerationClock=tic;
for point=find(~singleLeaf)
    pointClock=tic;
    t=tAll(:,point);radiusSquared=radiiSquared(point);
    total=-inf;maximum=-inf;used=0;leaves=0;budgetExceeded=false;
    innovation=zeros(dimension,1);
    visit(1,0);
    if budgetExceeded
        remaining=logGlobalPeak;
    else
        remaining=logGlobalPeak+.5*(dimension*(1+log(radiusSquared/dimension))-radiusSquared);
    end
    logDensity(point)=total-logJacobian;
    logError(point)=remaining-logJacobian;
    logMaximum(point)=maximum-logJacobian;
    expansionCount(point)=used;leafCount(point)=leaves;hitBudget(point)=budgetExceeded;
    pointSeconds(point)=toc(pointClock);
end
enumerationSeconds=toc(enumerationClock);
detail=struct('logErrorBound',logError,'relativeErrorBound',exp(logError-logDensity), ...
    'converged',~hitBudget & logError-logDensity<=log(tolerance)+1e-12, ...
    'logMaximumComponent',logMaximum,'maximumCertified',~hitBudget, ...
    'expansions',expansionCount,'leaves',leafCount,'radius',sqrt(radiiSquared),'hitBudget',hitBudget, ...
    'pointSeconds',pointSeconds,'setupSeconds',setupSeconds,'enumerationSeconds',enumerationSeconds, ...
    'seconds',toc(started),'singleLeafCount',sum(singleLeaf), ...
    'logGlobalUpper',logGlobalPeak-logJacobian, ...
    'logRemainingUpper',sum(logPeak(2:end))-sum(log(periods(2:end))));

    function visit(j,distance)
        conditionalCenter=root(j,1:j-1)*innovation(1:j-1,1)-t(j);
        width=sigma(j)*sqrt(max(0,radiusSquared-distance));
        integers=ceil(conditionalCenter-width):floor(conditionalCenter+width);
        [~,order]=sort(abs(integers-conditionalCenter));
        for integer=integers(order)
            if used>=maxExpansions
                budgetExceeded=true;return
            end
            used=used+1;
            innovation(j)=(integer-conditionalCenter)/sigma(j);
            nextDistance=distance+innovation(j)^2;
            if j==dimension
                value=logNormalizer-.5*nextDistance;
                total=add_logs(total,value);maximum=max(maximum,value);leaves=leaves+1;
            else
                visit(j+1,nextDistance);
                if budgetExceeded,return;end
            end
        end
    end
end

function value=add_logs(a,b)
if isinf(a) && a<0
    value=b;
elseif isinf(b) && b<0
    value=a;
else
    top=max(a,b);value=top+log1p(exp(min(a,b)-top));
end
end
