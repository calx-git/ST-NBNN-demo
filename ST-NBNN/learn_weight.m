function [weight, bias] = learn_weight(X, Y, options)

    model = train(Y, sparse(X), options);
    w = model.w;
    weight = w';
    bias = model.bias;