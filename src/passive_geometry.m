function [range,jacobian,hessian]=passive_geometry(position,transmitters,receivers)
sources=size(transmitters,1); nodes=size(receivers,1);
tx=repmat(transmitters,nodes,1); rx=repelem(receivers,sources,1);
deltaTx=position(:)'-tx; deltaRx=position(:)'-rx;
distanceTx=vecnorm(deltaTx,2,2); distanceRx=vecnorm(deltaRx,2,2);
range=distanceTx+distanceRx-vecnorm(tx-rx,2,2);
if nargout<2, return; end
unitTx=deltaTx./distanceTx; unitRx=deltaRx./distanceRx;
jacobian=unitTx+unitRx;
if nargout<3, return; end
hessian=zeros(2,2,numel(range));
for path=1:numel(range)
    a=unitTx(path,:)'; b=unitRx(path,:)';
    hessian(:,:,path)=(eye(2)-a*a')/distanceTx(path)+(eye(2)-b*b')/distanceRx(path);
end
end
