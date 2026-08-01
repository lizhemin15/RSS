# INR for image completion
# TIP paper reproducible example
import rss
import seaborn as sns
import matplotlib.pyplot as plt
import copy

for data_path in ['baboon_gray.png','boat.png','Cameraman.jpg','house.tif','jetplane.tif','lake.tif','livingroom.tif']:
    for mask_type in ['random']:
        parameters = {}
        net_list = []
        size = 256
        gpu_id = 0
        inrr_alpha = 0.2
        nabla_matrix_order_k = 2  # k for INRR+ (Table III default)
        lap_k = 1
        huber_delta = 0.2

        net_list.append({'net_name':"SIREN",'dim_in':2,'w0_initial':30,'dim_hidden':size})
        parameters['net_p'] = {'gpu_id':gpu_id,'net_name':'composition','net_list':net_list}
        parameters['data_p'] = {'data_shape':(size,size),'random_rate':0.5,
                                'pre_full':True,'mask_type':mask_type,'mask_path':'../mask/mask.png',
                                'data_path':'../img/'+data_path,'data_type':'gray_img','ymode':'completion',
                               'noise_mode':'gaussian','noise_parameter':25}

        parameters['train_p'] = {'train_epoch':2000}
        parameters['show_p'] = {'show_type':'gray_img','show_content':'original'}
        parameters['opt_p'] = {'net': {'opt_name': 'Adam', 'lr': 1e-3, 'weight_decay': 0},
                               'reg': {'opt_name': 'Adam', 'lr': 1e-4, 'weight_decay': 0},
                               'noise': {'opt_name': 'Adam', 'lr': 1e-4, 'weight_decay': 0}}

        # ---- Choose one regularization below ----

        # TV
        # parameters['reg_p'] = {'reg_name': 'TV', 'coef':1e-2,'p_norm': 1}

        # STV
        # parameters['reg_p'] = {'reg_name': 'STV', 'coef':2e-2,'p_norm': 1}

        # Huber TV
        # parameters['reg_p'] = {'reg_name': 'TV', 'coef':1e-2,'lap_mode':'Huber','huber_delta':0.1}

        # WTV
        # parameters['reg_p'] = {'reg_name': 'WTV', 'coef':1e-2}

        # AIR (row + col)
        # parameters['reg_p'] = {'reg_name':'MultiReg','reg_list':[{'reg_name':'AIR','coef':1e-3,'n':256,'mode':0},{'reg_name':'AIR','coef':1e-3,'n':256,'mode':1}]}

        # INRR+ (row + col, INRR with TV nabla blend)
        # parameters['reg_p'] = {'reg_name':'MultiReg','reg_list':[
        #     {'reg_name':'INRR','coef':1e-2,'n':size,'mode':0,'w0_initial':1.,'lap_k':lap_k,'inrr_alpha':inrr_alpha,'nabla_matrix_order_k':nabla_matrix_order_k},
        #     {'reg_name':'INRR','coef':1e-2,'n':size,'mode':1,'w0_initial':1.,'lap_k':lap_k,'inrr_alpha':inrr_alpha,'nabla_matrix_order_k':nabla_matrix_order_k}]}

        # INRR+ with Huber (row + col)
        parameters['reg_p'] = {'reg_name':'MultiReg','reg_list':[
            {'reg_name':'INRR','coef':1e-3,'n':size,'mode':0,
             'w0_initial':1.,'lap_k':lap_k,'lap_mode':'Huber',
             'huber_delta':huber_delta,'inrr_alpha':inrr_alpha,
             'nabla_matrix_order_k':nabla_matrix_order_k,
             'inr_parameter':{'dim_in': 1,'dim_out':32,'num_layers':5,'w0_initial':2.}},
            {'reg_name':'INRR','coef':1e-3,'n':size,'mode':1,
             'w0_initial':1.,'lap_k':lap_k,'lap_mode':'Huber',
             'huber_delta':huber_delta,'inrr_alpha':inrr_alpha,
             'nabla_matrix_order_k':nabla_matrix_order_k,
             'inr_parameter':{'dim_in': 1,'dim_out':32,'num_layers':5,'w0_initial':2.}}]}

        # INRR+ with logcosh (row + col)
        # parameters['reg_p'] = {'reg_name':'MultiReg','reg_list':[
        #     {'reg_name':'INRR','coef':1e-2,'n':size,'mode':0,'w0_initial':1.,'lap_k':lap_k,'lap_mode':'logcosh','huber_delta':0.3,'inrr_alpha':inrr_alpha,'nabla_matrix_order_k':nabla_matrix_order_k},
        #     {'reg_name':'INRR','coef':1e-2,'n':size,'mode':1,'w0_initial':1.,'lap_k':lap_k,'lap_mode':'logcosh','huber_delta':0.3,'inrr_alpha':inrr_alpha,'nabla_matrix_order_k':nabla_matrix_order_k}]}

        # INRR+ with quantile (row + col)
        # parameters['reg_p'] = {'reg_name':'MultiReg','reg_list':[
        #     {'reg_name':'INRR','coef':1e-2,'n':size,'mode':0,'w0_initial':1.,'lap_k':lap_k,'lap_mode':'quantile','quantile_q':0.5,'inrr_alpha':inrr_alpha},
        #     {'reg_name':'INRR','coef':1e-2,'n':size,'mode':1,'w0_initial':1.,'lap_k':lap_k,'lap_mode':'quantile','quantile_q':0.5,'inrr_alpha':inrr_alpha}]}

        # INRR+ with lp norm (row + col)
        # lp = 2
        # parameters['reg_p'] = {'reg_name':'MultiReg','reg_list':[
        #     {'reg_name':'INRR','coef':1e-2,'n':size,'mode':0,'w0_initial':1.,'lap_k':1,'lap_mode':'lp','norm_lap_lp':lp},
        #     {'reg_name':'INRR','coef':1e-2,'n':size,'mode':1,'w0_initial':1.,'lap_k':1,'lap_mode':'lp','norm_lap_lp':lp}]}

        # GroupReg (patch-level INRR with kmeans grouping)
        # parameters['reg_p'] = {'reg_name':'GroupReg','coef':1e-2,
        #     'group_para':{'n_clusters':22,'metric':'cosine','reg_mode':'single'},
        #     'each_reg_name':'INRR','start_epoch':100,'gpu_id':gpu_id,'w0_initial':1.,
        #     'x_trans':'patch','stride': 13,'patch_size':16,'search_epoch':1e2,'filter_type':None,'sigma':1e0,'lap_k':3}

        # GroupReg with Huber
        # parameters['reg_p'] = {'reg_name':'GroupReg','coef':1e-2,
        #     'group_para':{'n_clusters':22,'metric':'cosine','reg_mode':'single'},
        #     'each_reg_name':'INRR','start_epoch':100,'gpu_id':gpu_id,'w0_initial':1.,
        #     'x_trans':'patch','stride': 13,'patch_size':16,'search_epoch':1e2,'filter_type':None,'sigma':1e0,'lap_k':1,'lap_mode':'Huber','huber_delta':0.1}

        # ---- FRINR: replace SIREN with FRINR ----
        # To use FRINR, change net_list to:
        # net_list = [{'net_name':'FRINR','mode':'relu+fr','dim_in':2,'dim_hidden':256,'dim_out':1,'num_layers':4,
        #              'high_freq_num':128,'low_freq_num':128,'phi_num':32,'alpha':0.05}]
        # or for sin+fr mode:
        # net_list = [{'net_name':'FRINR','mode':'sin+fr','dim_in':2,'dim_hidden':256,'dim_out':1,'num_layers':4,
        #              'high_freq_num':128,'low_freq_num':128,'phi_num':32,'alpha':0.01,
        #              'first_omega_0':30.0,'hidden_omega_0':30.0}]

        parameters['save_p'] = {'save_if':False,'save_path':None}
        rssnet = rss.rssnet(parameters,verbose=False)

        rssnet.show_p['show_content'] = 'recovered'
        for i in range(10):
            rssnet.train(verbose=False)
            psnr_now = max(rssnet.log_dict['psnr'])
            nmae_now = min(rssnet.log_dict['nmae'])
            print(psnr_now,nmae_now)
