function [List]=assign_scores(Data,dim,priorEdge,priorNonEdge,delta,d,learnSetSize,stopEps)
%Data - list of edges, used to embed network into space; Mx2 matrix of
%edges
%dim - dimension used for embedding
%priorEdge - p(edge)
%priorNonEdge - p(nonedge)
%d - number of Gaussian mixtures to use for learning
%delta - distance cutoff; all pairs of nodes with distance >delta are
%considered to be non edges
%learnSetSize - how big should be the set size to learn p(dist|edge) and
%p(dist|nonedge)
%stopEps - stopping epsilon for EM algorithm

Loc=embed_novi(Data,dim); % dim-D coordinates of points in the embedding
D=dist(Loc'); % distance matrix

[N N]=size(D); % N - number of points in the network

%Lear p(dist|edge) and p(dist|nonedge)
[M a]=size(Data);
AdjMatr=zeros(N); % adjacency matrix
for i=1:M
    nodes=Data(i,1:2);
    AdjMatr(nodes(1),nodes(2))=1;
    AdjMatr(nodes(2),nodes(1))=1;
end

[Xedge Yedge]=find(AdjMatr==1);
ind=find(Xedge<Yedge);
Xedge=Xedge(ind);
Yedge=Yedge(ind);

[Xnonedge Ynonedge]=find(AdjMatr==0);
ind=find(Xnonedge<Ynonedge);
Xnonedge=Xnonedge(ind);
Ynonedge=Ynonedge(ind);

indEdge=ceil(rand(1,learnSetSize)*numel(Xedge));  
indNonEdge=ceil(rand(1,learnSetSize)*numel(Xnonedge));  

for i=1:numel(indEdge)
    EdgeDist(i)=D(Xedge(indEdge(i)),Yedge(indEdge(i)));
    NonEdgeDist(i)=D(Xnonedge(indNonEdge(i)),Ynonedge(indNonEdge(i)));
end

[pi1 mu1 s1 LL1 BIC1]=EM(EdgeDist,d,stopEps); % learn first distribution
[pi2 mu2 s2 LL2 BIC2]=EM(NonEdgeDist,d,stopEps); % learn second distribution
%p(dist|edge) and p(dist|nonedge) learned

%compute scores
[X Y]=find(D<=delta);
ind=find(X<Y);
X=X(ind);
Y=Y(ind);
List=zeros(numel(X),3);

for i=1:numel(X)
    List(i,1)=X(i);
    List(i,2)=Y(i);
    
    tmpEdge=0; %evidence
    tmpNonEdge=0; 
    for k=1:d
        tmpEdge=tmpEdge+pi1(k)*mvnpdf(D(X(i),Y(i)),mu1(k),s1(k));
        tmpNonEdge=tmpNonEdge+pi2(k)*mvnpdf(D(X(i),Y(i)),mu2(k),s2(k));
    end
    
    tmpEdge=tmpEdge*priorEdge; % posterior
    tmpNonEdge=tmpNonEdge*priorNonEdge; % posterior
    
    ProbEdge=tmpEdge/(tmpEdge+tmpNonEdge);
    tmpEdge+tmpNonEdge
    List(i,3)=ProbEdge; % degree of belief for pair to be an edge    
end
Max_score=max(List(:,3));
List(:,3)=List(:,3)/Max_score; % make max score equals to 1;



