rng('default');

 load('U_train5.mat');
 load('V_train5.mat');
 load('W_train5.mat');
 load('X_train5.mat');
 load('Y_train5.mat');
 load('Z_train5.mat');
 
 input_data=[U_train V_train W_train];
 target_data=[X_train Y_train Z_train];
 
 input_train_1=input_data(1:1:end, :);
 output_train_1= target_data(1:1:end, :);

 
 input_train= input_train_1';
 output_train= output_train_1';

 
net = newff(minmax(input_train),[3 23 50 23 3],{'purelin' 'purelin' 'poslin' 'purelin' 'purelin'},'trainlm');
net.trainParam.epochs = 1000;
net.trainParam.goal=0.00;
net.trainParam.lr = 0.01;
net.divideFcn = 'dividerand';        %# how to divide data
net.divideParam.trainRatio = 85/100; %# training set
net.divideParam.valRatio = 5/100;   %# validation set
net.divideParam.testRatio = 10/100;  %# testing set
% net.trainParam.max_fail=1; 
% net.trainParam.mem_reduc=1;
% net.trainParam.min_grad=1e-10;
% net.trainParam.mu=0.1;
% net.trainParam.mu_dec=0.010;
% net.trainParam.mu_inc=10; 
% net.trainParam.mu_max=1e10;
% net.trainParam.show=25;
% net.trainParam.time=inf;
[net,tr]=train(net,input_train,output_train);
grid
final = [0,0,0]
prev_y_test=[0;0;0]




 % *************************** ROS PART **************************
 
 
 pub_rc = rospublisher('/robot_coordinates' , 'geometry_msgs/Quaternion');
 msg_rc = rosmessage(pub_rc);
      msg_rc.X = 0;
      msg_rc.Y = 0;
      msg_rc.Z = 0; 
      msg_rc.W = 0;
      send(pub_rc,msg_rc);
 
  while 1
     
     sub=rossubscriber('/weld_line');
     msg = receive(sub,1000);
     a= msg.X; 
     b = msg.Y;
     c = msg.Z;
     input_test = [a;	b;	c];
     disp(input_test);
   
     
     y_test = sim(net,input_test);
    
     disp(y_test);
      
    if y_test ~= prev_y_test  
     
      msg_rc = rosmessage(pub_rc);
      msg_rc.X = y_test(1,1);
      msg_rc.Y = y_test(2,1);
      msg_rc.Z = y_test(3,1); 
      msg_rc.W = 0;
      send(pub_rec,msg_rc);
      end
      prev_y_test=y_test;
      t=y_test'
  end
 
view(net);


