function [logDensity,detail,moments] = wrapped_gaussian_box(residual,covariance,periods,tolerance,maxTerms,radius)
periods=periods(:);dimension=numel(periods);pointCount=size(residual,2);
terms=(2*radius+1)^dimension;
assert(terms<=maxTerms && radius>=0 && radius==floor(radius));
timer=tic;
scaledCovariance=covariance./(periods*periods');
root=chol(scaledCovariance,'lower');
axes=cell(1,dimension);[axes{:}]=ndgrid(-radius:radius);
lattice=zeros(dimension,terms);
for j=1:dimension,lattice(j,:)=axes{j}(:)';end
integerWhitened=root\lattice;
integerNorm=sum(integerWhitened.^2,1);
t=mod(residual./periods+.5,1)-.5;
whitened=root\t;
normalizer=-sum(log(diag(root)))-dimension/2*log(2*pi)-sum(log(periods));
logDensity=zeros(1,pointCount);logMaximum=zeros(1,pointCount);
if nargout>2
    meanResidual=zeros(dimension,pointCount);conditionalCovarianceSum=zeros(dimension);
end
blockSize=max(1,floor(2e6/terms));
for first=1:blockSize:pointCount
    indices=first:min(first+blockSize-1,pointCount);
    quadratic=sum(whitened(:,indices).^2,1)'+2*whitened(:,indices)'*integerWhitened+integerNorm;
    values=normalizer-.5*quadratic;
    peak=max(values,[],2);
    relativeWeights=exp(values-peak);
    totals=sum(relativeWeights,2);
    logDensity(indices)=(peak+log(totals))';
    logMaximum(indices)=peak';
    if nargout>2
        weights=relativeWeights./totals;
        latticeMean=lattice*weights';
        local=t(:,indices);meanNormalized=local+latticeMean;
        for row=1:dimension
            centeredRow=lattice(row,:)-latticeMean(row,:)';
            weightedRow=weights.*centeredRow;
            for column=row:dimension
                centeredColumn=lattice(column,:)-latticeMean(column,:)';
                value=sum(weightedRow.*centeredColumn,'all');
                conditionalCovarianceSum(row,column)=conditionalCovarianceSum(row,column)+value;
                if column~=row
                    conditionalCovarianceSum(column,row)=conditionalCovarianceSum(column,row)+value;
                end
            end
        end
        meanResidual(:,indices)=periods.*meanNormalized;
    end
end
diagonal=sum(abs(scaledCovariance),2);
majorization=.5*sum(log(diagonal))-sum(log(diag(root)));
peaks=zeros(dimension,1);conditionalPeaks=peaks;tails=zeros(dimension,pointCount);
for j=1:dimension
    [lower,errorBound]=wrapped_normal_interval(0,diagonal(j),1,1e-13);
    peaks(j)=add_logs(lower,errorBound);
    [lower,errorBound]=wrapped_normal_interval(0,root(j,j)^2,1,1e-13);
    conditionalPeaks(j)=add_logs(lower,errorBound);
    positive=scalar_tail(radius+1+t(j,:),diagonal(j));
    negative=scalar_tail(radius+1-t(j,:),diagonal(j));
    tails(j,:)=add_logs(positive,negative);
end
outside=tails+sum(peaks)-peaks;
top=max(outside,[],1);
logError=majorization+top+log(sum(exp(outside-top),1))-sum(log(periods));
globalUpper=sum(conditionalPeaks)-sum(log(periods));
logError=min(logError,globalUpper);
detail=struct('logErrorBound',logError,'relativeErrorBound',exp(logError-logDensity), ...
    'converged',logError-logDensity<=log(tolerance),'logMaximumComponent',logMaximum, ...
    'expansions',dimension*terms*ones(1,pointCount),'leaves',terms*ones(1,pointCount), ...
    'logGlobalUpper',globalUpper,'logRemainingUpper',sum(conditionalPeaks(2:end))-sum(log(periods(2:end))), ...
    'seconds',toc(timer),'radius',radius,'terms',terms,'largestValueBlock',min(blockSize,pointCount)*terms);
if nargout>2
    moments=struct('meanResidual',meanResidual,'locationScore',covariance\meanResidual, ...
        'meanConditionalCovariance',(periods*periods').*(conditionalCovarianceSum/pointCount));
end
end

function value=scalar_tail(distance,variance)
argument=distance/sqrt(2*variance);
value=add_logs(-argument.^2-.5*log(2*pi*variance),log(.5)+log(erfcx(argument))-argument.^2);
end

function value=add_logs(a,b)
top=max(a,b);value=top+log1p(exp(min(a,b)-top));
zero=isinf(top)&top<0;value(zero)=-inf;
end
