function setup = theta_lattice_setup(covariance)
clock = tic;
dimension = size(covariance,1);
omega = 1i*(covariance\eye(dimension))/(2*pi);
operations = cell(0,1);
svpCandidates = 0;
for iteration = 1:100
    transform = integer_lattice_reduction(chol(imag(omega)));
    omega = transform.'*omega*transform;
    omega = (omega+omega.')/2;
    operations{end+1} = struct('kind',"basis",'matrix',transform);
    [shortest,squaredLength,visited] = shortest_vector(chol(imag(omega)),2000000);
    svpCandidates = svpCandidates+visited;
    transform = primitive_completion(shortest);
    omega = transform.'*omega*transform;
    omega = (omega+omega.')/2;
    operations{end+1} = struct('kind',"basis",'matrix',transform);
    shift = round(real(omega));
    omega = omega-shift;
    operations{end+1} = struct('kind',"shift",'vector',diag(shift)/2);
    if abs(omega(1,1)) >= 1-32*eps
        break
    end
    pivot = omega(1,1);
    column = omega(2:end,1);
    operations{end+1} = struct('kind',"inverse",'pivot',pivot,'column',column);
    omega = [-1/pivot,column.'/pivot;column/pivot,omega(2:end,2:end)-column*column.'/pivot];
    omega = (omega+omega.')/2;
end
assert(iteration<100,'Theta:ReductionLimit','Siegel reduction did not terminate within its stated limit.');
assert(squaredLength>=sqrt(3)/2-1e-9,'Theta:ShortestVector','Reduced shortest-vector condition failed.');
setup = struct('omega',omega,'operations',{operations},'rho',sqrt(pi*squaredLength), ...
    'iterations',iteration,'svpCandidates',svpCandidates,'seconds',toc(clock));
end

function [bestVector,bestSquared,visited] = shortest_vector(upper,budget)
dimension = size(upper,1);
[bestSquared,index] = min(sum(upper.^2,1));
bestVector = zeros(dimension,1);
bestVector(index) = 1;
candidate = zeros(dimension,1);
visited = 0;
visit(dimension,0);

    function visit(level,distance)
        centre = -upper(level,level+1:end)*candidate(level+1:dimension,1)/upper(level,level);
        width = sqrt(max(0,bestSquared*(1+64*eps)-distance))/abs(upper(level,level));
        integers = ceil(centre-width):floor(centre+width);
        [~,order] = sort(abs(integers-centre));
        for integer=integers(order)
            visited = visited+1;
            assert(visited<=budget,'Theta:SVPBudget','Exact shortest-vector enumeration exhausted its setup budget.');
            candidate(level) = integer;
            next = distance+(upper(level,level)*(integer-centre))^2;
            if next>bestSquared*(1+64*eps),continue;end
            if level==1
                if any(candidate) && next<bestSquared
                    bestSquared = next;
                    bestVector = candidate;
                end
            else
                visit(level-1,next);
            end
        end
    end
end

function transform = primitive_completion(vector)
dimension = numel(vector);
rowTransform = eye(dimension);
current = vector;
for index=2:dimension
    first = current(1);second = current(index);
    if first==0 && second==0,continue;end
    [divisor,left,right] = gcd(first,second);
    block = [left,right;-second/divisor,first/divisor];
    rowTransform([1,index],:) = block*rowTransform([1,index],:);
    current([1,index]) = block*current([1,index]);
end
assert(current(1)==1 && all(current(2:end)==0),'Theta:PrimitiveVector','Shortest vector is not primitive.');
transform = round(rowTransform\eye(dimension));
assert(isequal(rowTransform*transform,eye(dimension)) && isequal(transform(:,1),vector), ...
    'Theta:UnimodularCompletion','Integer basis completion failed.');
end
