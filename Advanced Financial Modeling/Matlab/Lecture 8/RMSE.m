function z = RMSE(y, yhat)
% calculate root mean squared error
z = sqrt(mean((y-yhat).^2)); % Root Mean Sqaured Error
end