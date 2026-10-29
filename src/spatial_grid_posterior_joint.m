function result = spatial_grid_posterior_joint(coarse,fine,roi,spacing,transmitters,receivers,models,method,options,leafPoints,maxBlocks,seedPoints)
assert(isequal(size(roi),[2,2]) && all(isfinite(roi),'all') && all(diff(roi,1,2)>0));
assert(isscalar(spacing) && isfinite(spacing) && spacing>0);
assert(leafPoints>=1 && leafPoints==floor(leafPoints) && maxBlocks>=1 && maxBlocks==floor(maxBlocks));
assert(~isfield(options,'logPrior') && ~isfield(options,'maxQueries'));
assert(ismember(method,["marginal","maximum","difference","coarse"]));
counts=round(diff(roi,1,2)/spacing);
assert(all(counts>=1) && all(abs(counts*spacing-diff(roi,1,2))<=100*eps(max(abs(roi),[],2))));
pointCount=prod(counts);assert(pointCount<=flintmax);
assert(size(seedPoints,1)==2 && all(isfinite(seedPoints),'all'));
assert(all(seedPoints>=roi(:,1) & seedPoints<=roi(:,2),'all'));
seedIndices=zeros(2,0);
for seed=1:size(seedPoints,2)
    coordinate=(seedPoints(:,seed)-roi(:,1))/spacing+.5;
    [ix,iy]=ndgrid(unique([floor(coordinate(1)),ceil(coordinate(1))]), ...
        unique([floor(coordinate(2)),ceil(coordinate(2))]));
    seedIndices=[seedIndices,[ix(:)';iy(:)']];
end
seedIndices=max(1,min(counts,seedIndices));
seedIndices=unique(seedIndices','rows','stable')';nextSeed=1;

assert(isequal(size(coarse),size(fine)) && size(receivers,1)==size(coarse,2));
assert(size(transmitters,1)==size(coarse,1) && numel(models)==size(receivers,1));
started=tic;
nodeModels=prepare_bounds(coarse,fine,models,method);
pending=[1;counts(1);1;counts(2)];
[pendingUpper,boundLatticeExpansions]=block_upper(pending,roi,spacing,transmitters,receivers,nodeModels,method);
boundCalls=1;processed=0;queried=0;coveredPoints=0;
reportInterval=inf;lastReport=0;
if isfield(options,'progressSeconds') && options.progressSeconds>0
    reportInterval=options.progressSeconds;
end
expansions=0;leaves=0;unresolved=0;
leafBlocks=zeros(4,0);leafMeans=zeros(2,0);leafCovariances=zeros(2,2,0);
leafMaps=zeros(2,0);leafMaximum=zeros(1,0);leafSquaredMass=zeros(1,0);
leafLogLower=zeros(1,0);leafLogError=zeros(1,0);
lowerMass=-inf;queriedError=-inf;openMass=pendingUpper;tvBound=1;
localOptions=options;
localOptions.posteriorTolerance=options.posteriorTolerance/4;
while ~isempty(pendingUpper) && processed<maxBlocks && tvBound>options.posteriorTolerance
    selected=[];
    while nextSeed<=size(seedIndices,2)
        index=seedIndices(:,nextSeed);
        selected=find(pending(1,:)<=index(1) & pending(2,:)>=index(1) & ...
            pending(3,:)<=index(2) & pending(4,:)>=index(2),1);
        if ~isempty(selected),break;end
        nextSeed=nextSeed+1;
    end
    if isempty(selected),[~,selected]=max(pendingUpper);end
    parentUpper=pendingUpper(selected);
    block=pending(:,selected);
    pending(:,selected)=pending(:,end);pending(:,end)=[];
    pendingUpper(selected)=pendingUpper(end);pendingUpper(end)=[];
    blockCount=(block(2)-block(1)+1)*(block(4)-block(3)+1);
    if blockCount<=leafPoints
        [ix,iy]=ndgrid(block(1):block(2),block(3):block(4));
        points=[roi(1,1)+(ix(:)'-.5)*spacing;roi(2,1)+(iy(:)'-.5)*spacing];
        prediction=passive_geometry_grid(points,transmitters,receivers);
        local=finite_grid_posterior(coarse,fine,prediction,points,models,method,localOptions);
        assert(~local.noPositiveMass,'SpatialGrid:ZeroLeaf','A leaf has no finite retained mass.');
        logFraction=log(blockCount/pointCount);
        leafBlocks(:,end+1)=block;leafMeans(:,end+1)=local.mean;
        leafCovariances(:,:,end+1)=local.covariance;leafMaps(:,end+1)=local.map;
        leafMaximum(end+1)=max(local.posterior);leafSquaredMass(end+1)=sum(local.posterior.^2);
        leafLogLower(end+1)=local.logLowerMass+logFraction;
        leafLogError(end+1)=local.logRemainderMass+logFraction;
        lowerMass=log_sum(leafLogLower);queriedError=log_sum(leafLogError);
        queried=queried+local.queried;coveredPoints=coveredPoints+blockCount;
        expansions=expansions+local.latticeExpansions;leaves=leaves+local.latticeLeaves;
        unresolved=unresolved+local.unconvergedFactors;
    else
        children=repmat(block,1,2);
        if block(2)-block(1)>=block(4)-block(3)
            middle=floor((block(1)+block(2))/2);
            children(2,1)=middle;children(1,2)=middle+1;
        else
            middle=floor((block(3)+block(4))/2);
            children(4,1)=middle;children(3,2)=middle+1;
        end
        childCounts=(children(2,:)-children(1,:)+1).*(children(4,:)-children(3,:)+1);
        [childUpper,currentBoundExpansions]=block_upper(children,roi,spacing,transmitters,receivers,nodeModels,method);
        boundLatticeExpansions=boundLatticeExpansions+currentBoundExpansions;
        childUpper=min(childUpper,parentUpper-log(blockCount/pointCount));
        pending=[pending,children];pendingUpper=[pendingUpper,childUpper+log(childCounts/pointCount)];
        boundCalls=boundCalls+2;
    end
    processed=processed+1;
    openMass=log_sum(pendingUpper);
    remainderMass=log_sum([queriedError,openMass]);
    tvBound=exp(remainderMass-log_sum([lowerMass,remainderMass]));
    if toc(started)-lastReport>=reportInterval
        lastReport=toc(started);
        fprintf('Spatial blocks %d, point queries %d, bound expansions %d, TV %.3g, %.1f s.\n', ...
            processed,queried,boundLatticeExpansions,tvBound,lastReport);
    end
end
remainderMass=log_sum([queriedError,openMass]);
noPositiveMass=~isfinite(lowerMass);
posteriorMean=nan(2,1);covariance=nan(2);map=nan(2,1);maximumMass=nan;effectivePoints=nan;
leafWeights=zeros(size(leafLogLower));
if ~noPositiveMass
    leafWeights=exp(leafLogLower-max(leafLogLower));leafWeights=leafWeights/sum(leafWeights);
    posteriorMean=leafMeans*leafWeights';covariance=zeros(2);
    for leaf=1:numel(leafWeights)
        displacement=leafMeans(:,leaf)-posteriorMean;
        covariance=covariance+leafWeights(leaf)*(leafCovariances(:,:,leaf)+displacement*displacement');
    end
    [maximumMass,bestLeaf]=max(leafWeights.*leafMaximum);
    map=leafMaps(:,bestLeaf);effectivePoints=1/sum(leafWeights.^2.*leafSquaredMass);
end
result=struct('mean',posteriorMean,'covariance',covariance,'map',map, ...
    'maximumMass',maximumMass,'effectivePoints',effectivePoints, ...
    'tvBound',tvBound,'converged',~noPositiveMass && tvBound<=options.posteriorTolerance, ...
    'noPositiveMass',noPositiveMass,'queried',queried,'queryFraction',queried/pointCount, ...
    'seconds',toc(started),'pointCount',pointCount,'coveredPoints',coveredPoints, ...
    'processedBlocks',processed,'boundCalls',boundCalls,'boundLatticeExpansions',boundLatticeExpansions, ...
    'latticeExpansions',expansions, ...
    'latticeLeaves',leaves,'unconvergedFactors',unresolved,'leafBlocks',leafBlocks, ...
    'leafWeights',leafWeights,'leafMeans',leafMeans,'leafCovariances',leafCovariances, ...
    'leafLogLower',leafLogLower,'leafLogError',leafLogError,'pendingBlocks',pending, ...
    'pendingLogUpper',pendingUpper,'logLowerMass',lowerMass,'logRemainderMass',remainderMass);
end

function nodes=prepare_bounds(coarse,fine,models,method)
sources=size(coarse,1);nodes=cell(size(models));
for node=1:numel(models)
    model=models{node};
    if method=="difference"
        coarseLinear=model.noiseCoarseRoot\(eye(sources)-model.basis*model.flatCalibrationGain);
        coarseOffset=coarseLinear*coarse(:,node);
        fineLinear=eye(sources)-model.flatConditionalGain;
        fineOffset=fine(:,node)-model.flatConditionalGain*coarse(:,node);
        conditional=model.flatConditionalCovariance;
    else
        coarseLinear=model.coarseRoot\eye(sources);
        coarseOffset=coarseLinear*(coarse(:,node)-model.meanBias);
        fineLinear=eye(sources)-model.conditionalGain;
        fineOffset=fine(:,node)-model.meanBias-model.conditionalGain*(coarse(:,node)-model.meanBias);
        conditional=model.conditionalCovariance;
    end
    periods=model.periods;
    normalized=conditional./(periods*periods');root=chol(normalized,'lower');
    basis=root\eye(sources);
    transform=integer_lattice_reduction(basis);
    [rotation,latticeRoot]=qr(basis*transform,0);
    diagonal=sum(abs(normalized),2);
    globalUpper=zeros(sources,1);
    if method~="coarse" && method~="maximum"
        for source=1:sources
            [lower,error]=wrapped_normal_interval(0,diagonal(source),1,1e-12);
            globalUpper(source)=log_sum([lower,error]);
        end
    end
    nodes{node}=struct('coarseLinear',coarseLinear,'coarseOffset',coarseOffset, ...
        'fineLinear',fineLinear./periods,'fineOffset',fineOffset./periods, ...
        'diagonal',diagonal,'globalUpper',globalUpper, ...
        'whitenedLinear',root\(fineLinear./periods),'root',root, ...
        'latticeRoot',latticeRoot,'latticeOffset',rotation'*basis,'boundRotation',rotation, ...
        'logJacobian',sum(log(periods)), ...
        'logInflation',.5*sum(log(diagonal))-sum(log(diag(root)))-sum(log(periods)), ...
        'logNormalizer',-sum(log(diag(root)))-sources/2*log(2*pi)-sum(log(periods)));
end
end

function [upper,boundExpansions]=block_upper(blocks,roi,spacing,transmitters,receivers,nodes,method)
upper=zeros(1,size(blocks,2));boundExpansions=0;
for index=1:size(blocks,2)
    block=blocks(:,index);
    center=roi(:,1)+spacing*([block(1)+block(2);block(3)+block(4)]/2-.5);
    halfWidth=spacing*[block(2)-block(1);block(4)-block(3)]/2;
    sourceVectors=center'-transmitters;sourceDistances=vecnorm(sourceVectors,2,2);
    coarseUpper=0;componentLog=0;componentGradient=zeros(2,1);componentRemainder=0;
    selectedUpper=zeros(1,numel(nodes));otherUpper=zeros(1,numel(nodes));
    nodeCenters=zeros(size(transmitters,1),numel(nodes));nodeRadii=zeros(1,numel(nodes));
    jointAvailable=method~="coarse";
    for node=1:numel(nodes)
        model=nodes{node};receiver=receivers(node,:);
        receiverDistance=norm(center'-receiver);
        prediction=sourceDistances+receiverDistance-vecnorm(transmitters-receiver,2,2);
        coarseCenter=model.coarseOffset-model.coarseLinear*prediction;
        coarseRadius=geometry_range(model.coarseLinear,center,halfWidth,transmitters,receiver);
        currentCoarseUpper=-.5*sum(max(abs(coarseCenter)-coarseRadius,0).^2);
        upper(index)=upper(index)+currentCoarseUpper;
        coarseUpper=coarseUpper+currentCoarseUpper;
        if method=="coarse",continue;end
        fineCenter=model.fineOffset-model.fineLinear*prediction;
        fineRadius=geometry_range(model.fineLinear,center,halfWidth,transmitters,receiver);
        nearest=abs(mod(fineCenter+.5,1)-.5);
        distance=max(nearest-fineRadius,0);
        if method=="maximum"
            fineUpper=model.logNormalizer-.5*sum(distance.^2./model.diagonal);
        else
            fineUpper=0;
            for source=1:numel(fineCenter)
                fineUpper=fineUpper+periodic_interval_upper(fineCenter(source),fineRadius(source), ...
                    model.diagonal(source),model.globalUpper(source));
            end
            fineUpper=model.logInflation+fineUpper;
        end
        whitenedRadius=norm(geometry_range(model.whitenedLinear,center,halfWidth,transmitters,receiver));
        [correlatedUpper,expansions,selectedResidual,otherUpper(node)]= ...
            correlated_interval_upper(fineCenter,whitenedRadius,model,inf);
        nodeCenters(:,node)=fineCenter;nodeRadii(node)=whitenedRadius;
        boundExpansions=boundExpansions+expansions;
        fineUpper=min(fineUpper,correlatedUpper);
        upper(index)=upper(index)+fineUpper;
        [coarseJacobian,coarseRemainder]=geometry_taylor(model.coarseLinear,center,halfWidth,transmitters,receiver);
        [fineJacobian,fineRemainder]=geometry_taylor(model.whitenedLinear,center,halfWidth,transmitters,receiver);
        if ~isfinite(coarseRemainder+fineRemainder)
            jointAvailable=false;continue;
        end
        fineJacobian=model.boundRotation'*fineJacobian;
        fineGradient=fineJacobian'*selectedResidual;
        selectedLog=model.logNormalizer-.5*sum(selectedResidual.^2);
        selectedError=norm(selectedResidual)*fineRemainder;
        selectedUpper(node)=selectedLog+abs(fineGradient)'*halfWidth+selectedError;
        componentLog=componentLog-.5*sum(coarseCenter.^2)+selectedLog;
        componentGradient=componentGradient+coarseJacobian'*coarseCenter+fineGradient;
        componentRemainder=componentRemainder+norm(coarseCenter)*coarseRemainder+selectedError;
    end
    if jointAvailable
        jointSelected=componentLog+abs(componentGradient)'*halfWidth+componentRemainder;
        envelopeGap=max(0,coarseUpper+sum(selectedUpper)-jointSelected);
        for node=1:numel(nodes)
            desiredOther=selectedUpper(node)-envelopeGap+log(1e-6/(2*numel(nodes)));
            if otherUpper(node)>desiredOther
                [~,expansions,~,otherUpper(node)]=correlated_interval_upper( ...
                    nodeCenters(:,node),nodeRadii(node),nodes{node},desiredOther);
                boundExpansions=boundExpansions+expansions;
            end
        end
        if any(otherUpper==inf),continue;end
        logProduct=0;logOtherProduct=-inf;
        for node=1:numel(nodes)
            logOtherProduct=log_sum([logOtherProduct+log_sum([selectedUpper(node),otherUpper(node)]), ...
                logProduct+otherUpper(node)]);
            logProduct=logProduct+selectedUpper(node);
        end
        jointUpper=log_sum([jointSelected,coarseUpper+logOtherProduct]);
        upper(index)=min(upper(index),jointUpper);
    end
end
end

function [upper,expansions,selectedResidual,otherUpper]=correlated_interval_upper(center,radius,model,desiredTailLog)
dimension=numel(center);t=mod(center+.5,1)-.5;expansions=0;
offset=model.latticeOffset*t;root=model.latticeRoot;
candidate=zeros(dimension,1);
for level=dimension:-1:1
    babaiLocation=-(offset(level)+root(level,level+1:end)*candidate(level+1:end))/root(level,level);
    candidate(level)=round(babaiLocation);
end
bestCandidate=candidate;bestSquared=sum((offset+root*candidate).^2);closestUsed=0;
visit_closest(dimension,0);
candidate=bestCandidate;retainedCandidate=bestCandidate;selectedResidual=offset+root*bestCandidate;
babaiDistance=sum(selectedResidual.^2);otherUpper=inf;expansions=closestUsed;
distance=sqrt(babaiDistance);
if radius>=distance && radius>0
    upper=inf;return
end
if radius==0
    gamma=1;inflation=0;
else
    increment=radius/(distance-radius);gamma=1+increment;
    inflation=dimension/2*log1p(increment)+radius^2/(2*increment);
end
logPeak=0;
for level=1:dimension
    [lower,error]=wrapped_normal_interval(0,gamma*model.root(level,level)^2,1,1e-12);
    logPeak=logPeak+log_sum([lower,error]);
end
logPeak=logPeak-model.logJacobian;
normalizer=model.logNormalizer-dimension/2*log(gamma);
logBabai=normalizer-.5*babaiDistance/gamma;
target=min(log(1e-8)+logBabai-logPeak,desiredTailLog-inflation-logPeak);
lowerRadius=dimension;upperRadius=max(dimension+1,babaiDistance/gamma+1);
while .5*(dimension*(1+log(upperRadius/dimension))-upperRadius)>target
    upperRadius=2*upperRadius;
end
for iteration=1:45
    middle=(lowerRadius+upperRadius)/2;
    if .5*(dimension*(1+log(middle/dimension))-middle)>target
        lowerRadius=middle;
    else
        upperRadius=middle;
    end
end
sphereSquared=gamma*upperRadius;
total=-inf;otherTotal=-inf;used=0;hitBudget=false;
visit(dimension,0);
expansions=used+closestUsed;
if hitBudget
    upper=inflation+logPeak;otherUpper=upper;
else
    logTail=logPeak+.5*(dimension*(1+log(upperRadius/dimension))-upperRadius);
    upper=inflation+log_sum([total,logTail]);
    otherUpper=inflation+log_sum([otherTotal,logTail]);
end

    function visit_closest(level,partialDistance)
        location=-(offset(level)+root(level,level+1:end)*candidate(level+1:end))/root(level,level);
        width=sqrt(max(0,bestSquared-partialDistance))/abs(root(level,level));
        integers=ceil(location-width):floor(location+width);
        [~,order]=sort(abs(integers-location));
        for integer=integers(order)
            if closestUsed>=2000,return;end
            closestUsed=closestUsed+1;candidate(level)=integer;
            next=partialDistance+(root(level,level)*(integer-location))^2;
            if next>bestSquared,continue;end
            if level==1
                bestSquared=next;bestCandidate=candidate;
            else
                visit_closest(level-1,next);
            end
        end
    end

    function visit(level,partialDistance)
        location=-(offset(level)+root(level,level+1:end)*candidate(level+1:end))/root(level,level);
        width=sqrt(max(0,sphereSquared-partialDistance))/abs(root(level,level));
        integers=ceil(location-width):floor(location+width);
        [~,order]=sort(abs(integers-location));
        for integer=integers(order)
            if used>=2000,hitBudget=true;return;end
            used=used+1;candidate(level)=integer;
            next=partialDistance+(root(level,level)*(integer-location))^2;
            if next>sphereSquared,continue;end
            if level==1
                term=normalizer-.5*next/gamma;
                total=log_sum([total,term]);
                if any(candidate~=retainedCandidate)
                    otherTotal=log_sum([otherTotal,term]);
                end
            else
                visit(level-1,next);
                if hitBudget,return;end
            end
        end
    end
end

function [jacobian,remainder]=geometry_taylor(linear,center,halfWidth,transmitters,receiver)
sourceVectors=center'-transmitters;sourceDistances=vecnorm(sourceVectors,2,2);
receiverVector=center'-receiver;receiverDistance=norm(receiverVector);
sourceMinimum=vecnorm(max(abs(sourceVectors)-halfWidth',0),2,2);
receiverMinimum=norm(max(abs(receiverVector)-halfWidth',0));
jacobian=zeros(size(linear,1),2);remainder=inf;
if any(sourceMinimum==0) || receiverMinimum==0,return;end
jacobian=linear*(sourceVectors./sourceDistances+receiverVector/receiverDistance);
curvature=abs(linear)*(1./sourceMinimum)+abs(sum(linear,2))/receiverMinimum;
remainder=.5*sum(halfWidth.^2)*norm(curvature);
end

function bound=geometry_range(linear,center,halfWidth,transmitters,receiver)
radius=norm(halfWidth);
receiverCoefficient=abs(sum(linear,2));
bound=radius*(sum(abs(linear),2)+receiverCoefficient);
sourceMinimum=vecnorm(max(abs(transmitters-center')-halfWidth',0),2,2);
receiverMinimum=norm(max(abs(receiver-center')-halfWidth',0));
if all(sourceMinimum>0) && receiverMinimum>0
    sourceVectors=center'-transmitters;sourceDistances=vecnorm(sourceVectors,2,2);
    receiverVector=center'-receiver;
    jacobian=sourceVectors./sourceDistances+receiverVector/norm(receiverVector);
    curvature=abs(linear)*(1./sourceMinimum)+receiverCoefficient/receiverMinimum;
    local=abs(linear*jacobian)*halfWidth+.5*radius^2*curvature;
    bound=min(bound,local);
end
end

function upper=periodic_interval_upper(center,radius,variance,globalUpper)
if radius>=.5,upper=globalUpper;return;end
center=mod(center+.5,1)-.5;
normalizer=.5*log(2*pi*variance);a=1/(2*variance);
for width=0:100
    distances=max(abs(center+(-width:width))-radius,0);
    retained=log_sum(-a*distances.^2)-normalizer;
    threshold=sqrt(a)*(width+.5-radius);
    logIntegral=.5*log(pi)-log(2*sqrt(a))+log(erfcx(threshold))-threshold^2;
    tail=log(2)+log_sum([-threshold^2,logIntegral])-normalizer;
    upper=min(globalUpper,log_sum([retained,tail]));
    if tail-retained<=log(1e-10),return;end
end
error('SpatialGrid:SeriesBudget','The scalar interval bound exceeded its term budget.');
end

function value=log_sum(values)
if isempty(values),value=-inf;return;end
top=max(values);
if top==-inf,value=-inf;else,value=top+log(sum(exp(values-top)));end
end
