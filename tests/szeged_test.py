# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 17:29:08 2019

@author: JaguschJ
 """

import heat_flow_1d as hf

if __name__=='__main__':
    cover=hf.Material('cover',alpha=0,rho=1170,cp=1600,kh=0.35)
    #skim=hf.Material('skim',alpha=0, rho=1280,cp=1600,kh=0.35)
    skim=hf.Material('skim',alpha=0, rho=1280,cp=1600,kh=0.35)
    fabric=hf.Material('fabric',alpha=0, rho=1200,cp=1600,kh=0.15)

    if 0:
        
        print(cover.pprint())
        bot=hf.Layer('bottom',material=cover,thickness=4.e-3,initial_temp=20)
        
        s11=hf.Layer('skim1b',material=skim,thickness=0.175e-3,initial_temp=20)
        s12=hf.Layer('fabric1',material=fabric,thickness=0.7e-3,initial_temp=20)
        s13=hf.Layer('skim1t',material=skim,thickness=0.175e-3,initial_temp=20)
        
        s21=hf.Layer('skim2b',material=skim,thickness=0.175e-3,initial_temp=20)
        s22=hf.Layer('fabric2',material=fabric,thickness=0.7e-3,initial_temp=20)
        s23=hf.Layer('skim2t',material=skim,thickness=0.175e-3,initial_temp=20)
        
        s31=hf.Layer('skim3b',material=skim,thickness=0.175e-3,initial_temp=20)
        s32=hf.Layer('fabric3',material=fabric,thickness=0.7e-3,initial_temp=20)
        s33=hf.Layer('skim3t',material=skim,thickness=0.175e-3,initial_temp=20)
        
        s41=hf.Layer('skim4b',material=skim,thickness=0.175e-3,initial_temp=20)
        s42=hf.Layer('fabric4',material=fabric,thickness=0.7e-3,initial_temp=20)
        s43=hf.Layer('skim4t',material=skim,thickness=0.175e-3,initial_temp=20)
        topcover = hf.Layer('top',material=cover,thickness=6.e-3,initial_temp=20)
    if 1: 
        
        print(cover.pprint())
        bot=hf.Layer('bottom',material=cover,thickness=4.e-3,initial_temp=20)
        
        s11=hf.Layer('skim1b',material=skim,thickness=4*0.175e-3,initial_temp=20)
        s12=hf.Layer('fabric1',material=fabric,thickness=4*0.7e-3,initial_temp=20)
        s13=hf.Layer('skim1t',material=skim,thickness=4*0.175e-3,initial_temp=20)
        
#        s21=hf.Layer('skim2b',material=skim,thickness=0.175e-3,initial_temp=20)
#        s22=hf.Layer('fabric2',material=fabric,thickness=0.7e-3,initial_temp=20)
#        s23=hf.Layer('skim2t',material=skim,thickness=0.175e-3,initial_temp=20)
 #       
 #       s31=hf.Layer('skim3b',material=skim,thickness=0.175e-3,initial_temp=20)
 #       s32=hf.Layer('fabric3',material=fabric,thickness=0.7e-3,initial_temp=20)
 #       s33=hf.Layer('skim3t',material=skim,thickness=0.175e-3,initial_temp=20)
 #       
 #       s41=hf.Layer('skim4b',material=skim,thickness=0.175e-3,initial_temp=20)
 #       s42=hf.Layer('fabric4',material=fabric,thickness=0.7e-3,initial_temp=20)
 #       s43=hf.Layer('skim4t',material=skim,thickness=0.175e-3,initial_temp=20)
        topcover = hf.Layer('top',material=cover,thickness=6.e-3,initial_temp=20)
    #bot.n_points=13
    #top.n_points=19
    
    print(topcover.pprint())
    #L1.update(thickness=20)
    print(bot.pprint())
    topbc=hf.Boundary('top',timepoints=[0,2,120,1200],temperatures=[40,147,160,160],position=0)
    botbc=hf.Boundary('bot',timepoints=[0,2,120,1200],temperatures=[40,147,160,160],position=-1)
    print(topbc.pprint())
    print(botbc.pprint())

    belt=hf.Layout(default_temperature=90)
    print(belt.pprint())
    
    time,temp=belt.compute_temperature_field()
    print (belt.temperature_field)
    hf.plot_section(belt)
    #hf.plot2d(time,belt.x,belt.temperature_field)
    # coarsen results to speed up
    time_points,myfield = hf.Temperature_field(belt.x,belt.time_points,belt.temperature_field).func_trigger(1000)
    hf.plot2d(time_points,belt.x,myfield,boundary_ix=belt.layer_ix)
    hf.plot_fix_location(belt.x,time,temp,[belt.x[0],belt.x[24],belt.x[-1]])
    hf.plot_fix_time(belt.x,time,temp,[0,20,60,120,180,240,300])

