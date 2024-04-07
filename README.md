![](./statics/logo.jpg)

# Obj: RSSNet

## parameters
```
include:
1. net_parameters
2. noise_parameters
3. reg_parameters
4. data_parameters
5. task_parameters (cpu or gpu)

train_parameters
show_parameters
save_parameters
```



## Idea use way:  
```python
import rss
# parameters_dict = {}
# all the parameters are saved as json file
rssnet = rss.go(parameters_dict,parameters_dict_path=None)
# rss.go() include all of the following functions
rssnet = rss.net(net_p,noise_p,reg_p,data_p)
rssnet.train(train_p)
rssnet.show(show_p)
rssnet.save(save_p)
```