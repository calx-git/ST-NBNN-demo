function [weight, bias] = learn_weightT(X, Y, C, L, wp)

    D = size(X,2);
    N = size(Y,1);
    Cs = C*ones(1,N);

    Cs(find(Y==1)) = Cs(find(Y==1))*wp;


    cvx_begin quiet
        variables w(D) e(N) b
        minimize(0.5*w'*w + L*sum(Cs*e))

        subject to
    %         e >= 1 - Y.*(X*w + b)
            e >= pow_pos(1 - Y.*(X*w + b),2)
            e >= 0
            sum(w) == D
            w >= 0
    cvx_end

    weight = w;
    bias = b;