function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);
X= [ones(m,1)  X];
J=0;
s=y;

a1 = X;
a2 = sigmoid(a1*Theta1');
a2 = [ones(m,1) a2];
a3 = sigmoid(a2*Theta2');



yVec = repmat([1:num_labels], m, 1) == repmat(y, 1, num_labels);
%for i=1:num_labels,
%if(i==1),
%y = [y==i];
%else
%y = [y , s==i];
%end;
%end;

%for(j=1:num_labels),
%J(j,:) = -1/m *(y(j,:)' * prediction(:,j));
%end;

%J = sum(J);

z=y;
for j=1:num_labels,
for i = 1:m,
if z(i) == j,
y(i,j) = 1;
else
y(i,j)=0;
end;
end;
end;
yvec = y;


cost = -yVec .* log(a3) - (1 - yVec) .* log(1 - a3);

J = (1 / m) * sum(sum(cost));%without regularization


theta1ExcludingBias = Theta1(:, 2:end);
theta2ExcludingBias = Theta2(:, 2:end);

reg1 = sum(sum(theta1ExcludingBias .^ 2));
reg2 = sum(sum(theta2ExcludingBias .^ 2));

J = (1 / m) * sum(sum(cost)) + (lambda / (2 * m)) * (reg1 + reg2);%with regularization



delta1 = zeros(size(Theta1));
delta2 = zeros(size(Theta2));

for i=1:m,
d3 = a3(i,:)' - yVec(i,:)';

d2 = Theta2(:,2:end)'*d3.*sigmoidGradient(Theta1*a1(i,:)');
delta1 =  delta1 + (d2*a1(i,:));
delta2 =  delta2 + (d3*a2(i,:));
end;



c1 = [zeros(hidden_layer_size, 1)  Theta1(: ,2:end)];
c2 = [zeros(num_labels, 1)  Theta2(: ,2:end)];



Theta1_grad =  (1/m) *delta1 +(lambda/m)*c1;
 
Theta2_grad =  (1/m)*delta2 + (lambda/m)*c2;


grad = [Theta1_grad(:) ; Theta2_grad(:)];


end