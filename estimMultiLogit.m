function est = estimMultiLogit(Y,X)

[n,J] = size(Y);
nx = size(X,2);


% parameters of the Multinomial Logit
% b is a matrix with (nx) X (J-1);

function [ll,gr] = calclike(b)
    expv = [ones([n 1]) exp(X*b)];
    %pr = expv./repmat(sum(expv, 2), [1 j]);
    pr = bsxfun(@rdivide, expv,sum(expv, 2));
    ll = sum(log(pr(Y==1)));
    ll = -ll;
    gr = -X'*(Y(:,2:end) - pr(:,2:end));
end

opts = optimoptions('fminunc','MaxFunctionEvaluations',1e6,'MaxIterations',1e6,...
    'SpecifyObjectiveGradient',true, 'CheckGradients',true...
    , 'FiniteDifferenceType','central');
[b_MLE,~,~,~,~,H] = fminunc(@calclike, zeros([(nx) (J-1)]),opts);

est = struct();
est.b = b_MLE;
est.b_se = reshape(sqrt(diag(inv(H))),[nx (J-1)]);

end
