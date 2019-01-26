clear; 
clc;

load('U_train5.mat');
load('V_train5.mat');
load('W_train5.mat');
load('X_train5.mat');
load('Y_train5.mat');
load('Z_train5.mat');

input_data=[U_train V_train W_train];

target_data=[X_train Y_train Z_train];

input_train=input_data(1:1:45, :);
input_test=input_data(2:2:end, :);
 
output_train= target_data(1:1:45, :);
output_test= target_data(2:2:end, :);

input_train= input_train';
output_train= output_train';
input_test = input_test'
output_test = output_test'

%net = newff(input_train, output_train,[2 3 2],{'logsig' 'logsig' 'purelin'},'trainscg');
%net = newff([min(input_train(1,:)) max(input_train(1,:));min(input_train(2,:)) max(input_train(2,:));],[3],{'purelin'},'trainscg');
net = newff(minmax(input_train),[3 9 3],{'purelin' 'purelin' 'purelin'},'trainlm');
net.trainParam.epochs = 5000;
net.trainParam.goal=0;
net.trainParam.max_fail=1; 
net.trainParam.mem_reduc=1;
net.trainParam.min_grad=1e-10;
net.trainParam.mu=0.1;
net.trainParam.mu_dec=0.010;
net.trainParam.mu_inc=10; 
net.trainParam.mu_max=1e10;
net.trainParam.show=25;
net.trainParam.time=inf;
[net,tr]=train(net,input_train,output_train);
grid


while 1
    sub=rossubscriber('/centre_c');
    msg = receive(sub,1000);
    a= msg.X; 
    b = msg.Y;
    c = msg.Z;
    input_test = [a;	b;	c];
%output_test = [	578.5;-70.3	;750.5];
    y_test=sim(net,input_test);
    disp(y_test);
    pub_rc = rospublisher('/robot_coordinates' , 'geometry_msgs/Quaternion');
    msg_rc = rosmessage(pub_rc);
    msg_rc.X = y_test(1,1);
    msg_rc.Y = y_test(2,1);
    msg_rc.Z = y_test(3,1);
    msg_rc.W = 0;
    send(pub_rc,msg_rc);
end

%error = ((output_test-y_test)./y_test)*100
