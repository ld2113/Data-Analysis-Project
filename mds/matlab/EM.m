function [pi mu s LL BIC]=EM(x,d,eps)
%x - training data
%d - number of Gaussians
%eps - param to control the convergence

%initialization step
mu(d)=0; %means, size 1xd
s(d)=0; %sigmas, size 1xd
pi(d)=0; %coeficients, size 1xd

%initialize each mixture by choosing few points at random
for i=1:d
    r=ceil(rand(1,30)*100);
    mean=0;
    std=0;
    for j=1:30
        mean=mean+x(r(j));
    end
    mean=mean/30;
    for j=1:30
        std=std+(x(r(j))-mean)^2;
    end
    std=std/29;
    mu(i)=mean;
    s(i)=std;
    pi(i)=1/d; % this initialized uniformly for now
end
[m n]=size(x);
gama(n,d)=0; % responsibilities
%end of initialization

change=10000000;
oldLL=-1000000;
iter=1;
while change>=eps
    %E step
    for i=1:n
        norm_const=0; %normalization constant
        for k=1:d
            gama(i,k)=pi(k)*mvnpdf(x(i),mu(k),s(k));
            norm_const=norm_const+gama(i,k);
        end
        for k=1:d
            gama(i,k)=gama(i,k)/norm_const;
        end
    end    
    %end of E step
    %M step    
    for k=1:d
        Nk=0;
        Muk_new=0;
        for i=1:n
            Nk=Nk+gama(i,k);
            Muk_new=Muk_new+gama(i,k)*x(i);
        end
        Muk_new=Muk_new/Nk;
        mu(k)=Muk_new;
        Sigma_k_new=0;
        for i=1:n
            Sigma_k_new=Sigma_k_new+gama(i,k)*(x(i)-Muk_new)^2;
        end
        Sigma_k_new=Sigma_k_new/Nk;
        s(k)=Sigma_k_new;
        pi(k)=Nk/n;        
    end
        %end of M step
    if (rem(iter,10))==0
        %calculate log-likelihood 
        LL=0;
        for i=1:n
            sublog=0;
            for k=1:d
                sublog=sublog+pi(k)*mvnpdf(x(i),mu(k),s(k));
            end
            LL=LL+log(sublog);
        end
        change=abs(LL-oldLL);
        oldLL=LL;
        LL
    end
    iter=iter+1;        
end %end of big while
BIC=LL-(d/2)*log(n);