function result = continuous_range_baseline(observed,transmitters,receivers,models,roi,method)
timer=tic;
sources=size(transmitters,1);nodes=size(receivers,1);
whitening=cell(1,nodes);
for node=1:nodes
    model=models{node};
    if method=="difference"
        root=chol(model.noiseCovariance(1:sources,1:sources),'lower');
        [Q,~]=qr(root\model.basis);
        whitening{node}=Q(:,size(model.basis,2)+1:end)'/root;
    else
        whitening{node}=model.coarseRoot\eye(sources);
    end
end
[gx,gy]=ndgrid(linspace(roi(1,1),roi(1,2),7),linspace(roi(2,1),roi(2,2),7));
starts=[gx(:)';gy(:)'];values=zeros(1,size(starts,2));
for j=1:numel(values),values(j)=objective(starts(:,j));end
[~,order]=sort(values);starts=starts(:,order(1:9));
solutions=zeros(2,9);costs=zeros(1,9);iterations=zeros(1,9);stationarity=zeros(1,9);
stopReasons=repmat("iteration_limit",1,9);
for start=1:9
    position=starts(:,start);
    for iteration=1:100
        [~,gradient,hessian,residuals]=objective(position);
        assert(rcond(hessian)>1e-13,'The range baseline is locally unidentifiable under this contrast.');
        projected=position-min(roi(:,2),max(roi(:,1),position-gradient/norm(hessian,2)));
        if norm(projected)<1e-5
            stopReasons(start)="projected_gradient";
            break
        end
        step=box_quadratic_step(gradient,hessian,roi(:,1)-position,roi(:,2)-position);
        if gradient'*step>=0
            stopReasons(start)="non_descent_step";
            break
        end
        accepted=false;
        for reduction=0:31
            scale=2^-reduction;
            candidate=min(roi(:,2),max(roi(:,1),position+scale*step));
            costChange=objective_change(position,candidate,residuals);
            if costChange<=1e-4*gradient'*(candidate-position)
                accepted=true;
                break
            end
        end
        if ~accepted
            stopReasons(start)="line_search_limit";
            break
        end
        position=candidate;
    end
    [cost,gradient,hessian]=objective(position);
    solutions(:,start)=position;costs(start)=cost;iterations(start)=iteration;
    stationarity(start)=norm(position-min(roi(:,2),max(roi(:,1),position-gradient/norm(hessian,2))));
    if stationarity(start)<1e-5,stopReasons(start)="projected_gradient";end
end
[cost,best]=min(costs);
result=struct('estimate',solutions(:,best),'cost',cost,'solutions',solutions,'costs',costs, ...
    'iterations',iterations,'projectedGradient',stationarity,'seconds',toc(timer), ...
    'converged',stationarity(best)<1e-5,'objective',method,'stopReasons',stopReasons, ...
    'solver',"box_quadratic_Gauss_Newton_v3");

    function [cost,gradient,gaussNewton,residuals]=objective(position)
        [prediction,jacobian]=passive_geometry(position,transmitters,receivers);
        cost=0;gradient=zeros(2,1);gaussNewton=zeros(2);residuals=cell(1,nodes);
        for n=1:nodes
            rows=(n-1)*sources+(1:sources);
            residual=whitening{n}*(observed(:,n)-prediction(rows)-models{n}.meanBias);
            J=whitening{n}*jacobian(rows,:);residuals{n}=residual;
            cost=cost+.5*(residual'*residual);
            gradient=gradient-J'*residual;
            gaussNewton=gaussNewton+J'*J;
        end
    end

    function change=objective_change(position,candidate,residuals)
        displacement=(candidate-position)';squared=sum(displacement.^2);
        sourceVectors=position'-transmitters;receiverVectors=position'-receivers;
        sourceChange=(2*sum(sourceVectors.*displacement,2)+squared)./ ...
            (vecnorm(sourceVectors+displacement,2,2)+vecnorm(sourceVectors,2,2));
        receiverChange=(2*sum(receiverVectors.*displacement,2)+squared)./ ...
            (vecnorm(receiverVectors+displacement,2,2)+vecnorm(receiverVectors,2,2));
        rangeChange=sourceChange+receiverChange';
        change=0;
        for n=1:nodes
            delta=whitening{n}*rangeChange(:,n);
            change=change+.5*(delta'*delta)-residuals{n}'*delta;
        end
    end
end

function step=box_quadratic_step(gradient,hessian,lower,upper)
candidates=zeros(2,5);
column=1;
for fixed=1:2
    free=3-fixed;
    for boundary=[lower(fixed),upper(fixed)]
        column=column+1;
        candidates(fixed,column)=boundary;
        freeValue=-(gradient(free)+hessian(free,fixed)*boundary)/hessian(free,free);
        candidates(free,column)=min(upper(free),max(lower(free),freeValue));
    end
end
interior=-(hessian\gradient);
if all(interior>=lower & interior<=upper),candidates(:,end+1)=interior;end
values=gradient'*candidates+.5*sum(candidates.*(hessian*candidates),1);
[~,best]=min(values);
step=candidates(:,best);
end
