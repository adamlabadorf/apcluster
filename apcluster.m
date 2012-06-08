clear;
% affinity propagation clustering algorithm

% random data for testing
% X = zeros(100,10);
% X(1:20,:) = 3+rand(20,10)-0.5;
% X(21:50,:) = 15+rand(30,10)-0.5;
% X(51:100,:) = 9+rand(50,10)-0.5;
% 
% N = size(X,1);
% S_orig = zeros(N,N);
% for i=1:N
%    for j=i:N
%         diff = X(i,:)-X(j,:);
%         d = -norm(diff)^2;
%         S_orig(i,j) = d;
%         S_orig(j,i) = d;
%     end;
% end;
% S_orig = S_orig+eye(size(S_orig))*-1.25;

% motif data w/ 2 clusters
S_orig = importdata('js_div.txt');
self_sim_ratio = 0.1;
self_sim = (max(max(S_orig))+min(min(S_orig)))*self_sim_ratio;
S_orig = -S_orig;
S_orig = S_orig+eye(size(S_orig))*-self_sim;
X = S_orig;


S = S_orig;

N=size(S,1); A=zeros(N,N); R=zeros(N,N); % Initialize messages
S=S+(eps*S+realmin*100).*rand(N,N); % Remove degeneracies
lam=0.5; % Set damping factor
for iter=1:100
    % Compute responsibilities
    Rold=R;
    AS=A+S;
    [Y,I]=max(AS,[],2);
    for i=1:N 
        AS(i,I(i))=-realmax;
    end;
    [Y2,I2]=max(AS,[],2);
    R=S-repmat(Y,[1,N]);
    for i=1:N 
        R(i,I(i))=S(i,I(i))-Y2(i); 
    end;
    R=(1-lam)*R+lam*Rold; % Dampen responsibilities

    % Compute availabilities
    Aold=A;
    Rp=max(R,0); 
    for k=1:N 
        Rp(k,k)=R(k,k); 
    end;
    A=repmat(sum(Rp,1),[N,1])-Rp;
    dA=diag(A); A=min(A,0); 
    for k=1:N 
        A(k,k)=dA(k); 
    end;
    A=(1-lam)*A+lam*Aold; % Dampen availabilities
end;
E=R+A; % Pseudomarginals
end_I=find(diag(E)>0); K=length(end_I); % Indices of exemplars
[tmp c]=max(S(:,end_I),[],2);
c(end_I)=1:K;
idx=end_I(c); % Assignments

subplot(1,2,1);
pcolor(repmat(idx,1,2));
colorbar();
subplot(1,2,2);
pcolor(X);
colorbar();