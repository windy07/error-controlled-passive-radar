function result=profile_cancelled_caf(reference,surveillance,lagLimits)
count=numel(reference);oversample=8;shifts=-4:4;
assert(mod(count,2)==0 && isequal(size(reference),size(surveillance)));
templates=zeros(count,numel(shifts),'like',reference);
for index=1:numel(shifts),templates(:,index)=circshift(reference,shifts(index));end
[basis,~]=qr(templates,0);
projected=surveillance-basis*(basis'*surveillance);
referenceSpectrum=fft(reference);
spectrum=fft([projected,basis],[],1).*conj(referenceSpectrum);
correlations=ifft([spectrum(1:count/2,:);zeros((oversample-1)*count,10);spectrum(count/2+1:end,:)],[],1)*oversample;
lags=(oversample*lagLimits(1):oversample*lagLimits(2))'/oversample;
sampled=correlations(mod(round(oversample*lags),oversample*count)+1,:);
energy=sum(abs(reference).^2);
numerator=abs(sampled(:,1)).^2;
denominator=energy-sum(abs(sampled(:,2:end)).^2,2);
singular=lags==round(lags) & abs(lags)<=4;
frequencies=[0:count/2-1,-count/2:-1]'/count;
stable=find(~singular & denominator<=1e-8*energy);
for index=stable'
    shifted=ifft(referenceSpectrum.*exp(-2i*pi*frequencies*lags(index)));
    residual=shifted-basis*(basis'*shifted);
    denominator(index)=sum(abs(residual).^2);
    numerator(index)=abs(residual'*surveillance)^2;
end
assert(all(denominator(~singular)>0));
score=zeros(size(lags));score(~singular)=numerator(~singular)./denominator(~singular);
[~,best]=max(score);[~,rawBest]=max(numerator);
result=struct('lags',lags,'numerator',numerator,'denominator',denominator, ...
    'profileScore',score,'singular',singular,'stableRecomputations',numel(stable), ...
    'referenceEnergy',energy,'projectedEnergy',sum(abs(projected).^2), ...
    'profileLag',lags(best),'numeratorLag',lags(rawBest),'profileIndex',best, ...
    'numeratorIndex',rawBest,'nuisanceShifts',shifts,'oversample',oversample);
end
