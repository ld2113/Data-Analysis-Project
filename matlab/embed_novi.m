function locations=embed_novi(A,dimen) %A=edges
% input: edges = list of pairs of nodes (matrix Nx2, of numbers 1:M)
% output: list of locations of the points (M x dimen)

randn('state',100)


mx=max(max(A));
A=[A; mx mx];
C=[A ones(size(A,1),1)];

Geom=spconvert(C);
Geom=sign(Geom+Geom');

Geom(mx,mx)=0;

%%%%%%%%%%%%%%%%%%%%%%%%%%

symcheck = max(max(Geom-Geom'));
Geom = Geom - diag(diag(Geom));
N = max(size(Geom));


%%%%%%%%%% Pathlengths up to K-1 %%%%%%%%
K = 30;
Kmax = 200;

Path = Geom;
Q = Path;

for k = 2:K-1,
    Q = Q*Geom;        % Geom to the power k
    Path = Path + ( (Path == 0) & (Q > 0) )*k;
end

P = Path - Kmax;
P = P.*(P~=-Kmax);
P = sparse(P);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% Des' subspace iteration %%%%%%%%%

psum = sum(P)';
psumsum = sum(sum(P));


vdiff = ones(dimen,1);

vdiffmx = 1;

count = 0;

for i=1:dimen
    v{i} = randn(N,1);   % random starting vectors
end



while vdiffmx > 1e-3

    for i=1:dimen
        mv(i) = mean(v{i});

        vnew{i} = -0.5*(P*v{i}  - mv(i)*psum + (mv(i)*psumsum/N - (psum'*v{i})*(1/N))*ones(N,1));
    end


    vnew{1} = vnew{1}/norm(vnew{1},2);

    for i=2:dimen
        pom=zeros(size(vnew{i}));

        for j=1:i-1
            pom=pom+(vnew{j}'*vnew{i})*vnew{j};
        end

        vnew{i}=vnew{i}-pom;
        vnew{i}=vnew{i}/norm(vnew{i},2);

    end

       count = count+1;


    for i=1:dimen
        vdiff(i) = norm(v{i} - vnew{i});
    end

       vdiffmx = max(vdiff);%tu

    for i=1:dimen
        v{i} = vnew{i};
    end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% Estimate eigenvalues %%%%%%%%%%%%%%%%%%%%%

for i=1:dimen
    mv(i) = mean(v{i});

    Av{i} = -0.5*(P*v{i}  - mv(i)*psum + (mv(i)*psumsum/N - (psum'*v{i})*(1/N))*ones(N,1));
    lam(i) = v{i}'*Av{i};

    xvals{i} = sqrt(lam(i))*v{i};
    xvals{i} = xvals{i}-min(xvals{i});
    xvals{i} = xvals{i}/max(xvals{i});
end

lam;

locations = xvals{1};

for i=2:dimen
    locations = [locations xvals{i}];
end
