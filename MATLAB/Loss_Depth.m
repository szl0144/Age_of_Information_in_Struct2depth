clear;

filename = "/Users/shaoyi/Downloads/loss_new.txt";  %this is the path to read txt file

[valid_loss,train_loss,time] = textread(filename,'%f %f %f');


figure

plot(time,train_loss)
title('Loss vs AoI (new)','FontSize',20)
xlabel('AoI','FontSize',20);
ylabel('Training Train Loss','FontSize',20);
xlim([0, 1.1]);
ylim([0, 0.3]);
figure
plot(time,valid_loss)
title('Valid Loss vs AoI','FontSize',20)
xlabel('AoI','FontSize',20);
ylabel('Valid Loss','FontSize',20);
xlim([0, 1.1]);
ylim([0, 0.4]);
