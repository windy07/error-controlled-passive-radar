function [logMaximum,detail]=wrapped_gaussian_maximum(residual,covariance,periods,~,maxExpansions)
setupClock=tic;periods=periods(:);dimension=numel(periods);
root=chol(covariance./(periods*periods'),'lower');
basis=root\eye(dimension);
transform=integer_lattice_reduction(basis);
[orthogonal,upper]=qr(basis*transform,0);
tAll=mod(residual./periods+.5,1)-.5;
offsetAll=orthogonal'*(basis*tAll);
normalizer=-sum(log(diag(root)))-dimension/2*log(2*pi)-sum(log(periods));
pointCount=size(residual,2);logMaximum=zeros(1,pointCount);logError=-inf(1,pointCount);
visits=zeros(1,pointCount);leaves=visits;pointSeconds=visits;hitBudget=false(1,pointCount);
integerOffsets=zeros(dimension,pointCount);
setupSeconds=toc(setupClock);
for point=1:pointCount
    clock=tic;offset=offsetAll(:,point);candidate=zeros(dimension,1);
    for coordinate=dimension:-1:1
        babaiCentre=-(offset(coordinate)+upper(coordinate,coordinate+1:end)* ...
            candidate(coordinate+1:dimension,1))/upper(coordinate,coordinate);
        candidate(coordinate)=round(babaiCentre);
    end
    bestCandidate=candidate;bestDistance=sum((offset+upper*candidate).^2);
    used=0;leafCount=0;exceeded=false;
    visit(dimension,0);
    integerOffsets(:,point)=transform*bestCandidate;
    logMaximum(point)=normalizer-.5*bestDistance;
    if exceeded,logError(point)=normalizer;end
    visits(point)=used;leaves(point)=leafCount;hitBudget(point)=exceeded;pointSeconds(point)=toc(clock);
end
detail=struct('logMaximumComponent',logMaximum,'logErrorBound',logError, ...
    'relativeErrorBound',exp(logError-logMaximum),'converged',~hitBudget,'maximumCertified',~hitBudget, ...
    'expansions',visits,'leaves',leaves,'hitBudget',hitBudget,'pointSeconds',pointSeconds, ...
    'setupSeconds',setupSeconds,'logGlobalUpper',normalizer,'integerOffsets',integerOffsets, ...
    'basisTransform',transform);

    function visit(level,distance)
        centre=-(offset(level)+upper(level,level+1:end)*candidate(level+1:dimension,1))/upper(level,level);
        width=sqrt(max(0,bestDistance*(1+64*eps)-distance))/abs(upper(level,level));
        integers=ceil(centre-width):floor(centre+width);
        [~,order]=sort(abs(integers-centre));
        for integer=integers(order)
            if used>=maxExpansions,exceeded=true;return;end
            used=used+1;candidate(level)=integer;
            next=distance+(upper(level,level)*(integer-centre))^2;
            if next>bestDistance*(1+64*eps),continue;end
            if level==1
                leafCount=leafCount+1;
                if next<bestDistance,bestDistance=next;bestCandidate=candidate;end
            else
                visit(level-1,next);
                if exceeded,return;end
            end
        end
    end
end
