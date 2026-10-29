function [logLower,logError,terms]=wrapped_normal_interval(residual,variance,period,tolerance)
shape=size(residual); residual=mod(residual(:)+period/2,period)-period/2;
if sqrt(variance)/period<.25
    a=period^2/(2*variance);
    normalizer=.5*log(2*pi*variance);
    for radius=0:100
        exponents=-(residual+period*(-radius:radius)).^2/(2*variance);
        center=max(exponents,[],2); logLower=center+log(sum(exp(exponents-center),2))-normalizer;
        t=sqrt(a)*(radius+.5);
        logIntegral=.5*log(pi)-log(2*sqrt(a))+log(erfcx(t))-t^2;
        logError=log(2)+log_add(-t^2,logIntegral)-normalizer;
        if all(logError-logLower<=log(tolerance))
            terms=2*radius+1; logLower=reshape(logLower,shape);
            logError=repmat(logError,shape); return
        end
    end
else
    a=2*pi^2*variance/period^2;
    for radius=0:100
        frequencies=1:radius;
        approximation=(1+2*sum(exp(-a*frequencies.^2).*cos(2*pi*residual*frequencies/period),2))/period;
        t=sqrt(a)*(radius+1);
        logIntegral=.5*log(pi)-log(2*sqrt(a))+log(erfcx(t))-t^2;
        logRemainder=log(2/period)+log_add(-t^2,logIntegral);
        lower=approximation-exp(logRemainder);
        if all(lower>0)
            logLower=log(lower); logError=log(2)+logRemainder;
            if all(logError-logLower<=log(tolerance))
                terms=radius+1; logLower=reshape(logLower,shape);
                logError=repmat(logError,shape); return
            end
        end
    end
end
error('WrappedNormal:SeriesBudget','One-dimensional series did not reach its truncation tolerance.');
end

function out=log_add(a,b)
center=max(a,b); out=center+log(exp(a-center)+exp(b-center));
end
