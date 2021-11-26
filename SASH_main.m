clc; clear all; close all;
addpath('.\evals\');
addpath('.\FOptM-share\');
addpath('.\utils\');
% Loading the data.
db_name = 'CIFAR-10'; 
nbits = 64; % number of bits of the hash codes.


load('CIFAR-10.mat');
X=X_tr';
Xtest = X_te';
Y = train_one_hot_sum';
trainlabel= trainlabel;
testlabel = testlabel;


%% Setting the parameters
Ntrain = size(X,2);
% Formation of the kernel matrix (PhiX) from anchors and X.
% Get anchors
n_anchors = 1000; 
rand('seed',1);
% Anchors are randomly sampled from the training dataset.
anchor = X(:,randsample(Ntrain,n_anchors));
% Set the experimental parameters as given in the paper
sigma = 1; 

alpha = 2e1;
beta = 1e1;
gamma = 1e-3;
lambda = 1e-1;

%% Phi of training data.
Dis = EuDist2(X',anchor',0);
bandwidth = mean(mean(Dis)).^0.5;
clear Dis;
PhiX = exp(-sqdist(X',anchor')/(bandwidth*bandwidth));
PhiX = [PhiX, ones(Ntrain,1)];
PhiX = PhiX';


%% Optimization Parameters
debug = 1;
tol = 1e-5;
% Init B
randn('seed',3);
B = sign(randn(Ntrain,nbits));
B = B';
c = size(Y,1); % size of the semantic alignment matrix
n = size(Y,2);

% Init S
S = (Y' * Y)/n;
M = zeros(n,n);
H = zeros(n,n);

% Init P
P = randn(size(PhiX,1),nbits);

%% Optimization
maxItr = 10;

for i = 1: maxItr    
    if debug,fprintf('Iteration  %03d: ',i);end
    P0 = P;
    % P-step
    D=diag(sum(S,2));% Init L
    L = D-S;
    P = pinv((PhiX*PhiX' + (gamma/lambda)*eye(size(PhiX,1)) +  (1/lambda)*PhiX*L*PhiX'))*PhiX*B';

    
    
    % S-step
    % get H,M
    for time1 = 1:n
        for time2 = 1:n
            H(time1,time2) = (norm(P'*PhiX(:,time1) - P'*PhiX(:,time2),2))^2 + beta*(norm(Y(:,time1) - Y(:,time2),2))^2;
            M(time1,time2) = (-2/(2*alpha)) * H(time1,time2);
        end
        S(time1,:) = SimplexProj(M(time1,:));
    end
    
    
    % B-step
    B = sign(P'*PhiX);

    
    bias = norm(B - P'*PhiX,'fro');
    if debug, fprintf('  bias=%g\n',bias);
    end
    
    if bias < tol*norm(B,'fro')
            break;
    end 
    if norm(P-P0,'fro') < tol * norm(P0)
        break;
    end

end

%%
% Phi of testing data.
Phi_Xtest = exp(-sqdist(Xtest',anchor')/(bandwidth*bandwidth));
Phi_Xtest = [Phi_Xtest, ones(size(Phi_Xtest,1),1)];
Phi_Xtest = Phi_Xtest';


%%
% Evaluation
display('Evaluation starts ...');
Xre = X_re';
Phi_Xre = exp(-sqdist(Xre',anchor')/(bandwidth*bandwidth));
Phi_Xre = [Phi_Xre, ones(size(Phi_Xre,1),1)];
Phi_Xre = Phi_Xre';
    
I_te = P'*Phi_Xtest > 0;
I_tr = P'*Phi_Xre > 0;
I_te = I_te';
I_tr = I_tr';
    
n_re = size(I_tr,1);
    
   
B = compactbit(I_tr);
tB = compactbit(I_te);
hammTrainTest = hammingDist(tB, B)';
    
L_te = test_one_hot_sum;
L_re = re_one_hot_sum;

mAP5000 = calmAP(L_te,L_re,hammTrainTest',5000);
mAPall = calmAP(L_te,L_re,hammTrainTest',n_re);
    

