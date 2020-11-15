#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 30 Sep 2017

@author : Simon Gottwald
@support: no active support
@contact: Jens.Jagusch@Contitech.De
"""
#Importing needed packages
import numpy as np
from scipy.integrate import odeint, ode
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import yaml
import sys
import os
import time as tm
#import vulkanisation as vc


class Layout(object):
    
    dictofMaterials={}
    listofLayers=[]
    listofBC=[]  # Boundaries (at least inner and outer or top/bot)
    def __init__(self,name='layout',delta_x=1,delta_t=1,default_temperature=20):
        '''
        delta_x  float
            distance of integration point in length
        delta_t   float
            distance of time integration points
        te_top object
            Boundary condition: list of pairs with (time,temperature) or 2dim numpy array
        te_bot object
            Boundary condition: list of pairs with (time,temperature) or 2dim numpy array
        '''
        self.name=name
        self.default_temperature=default_temperature
        assert delta_x > 0,'point distance must be > 0'
        assert delta_t > 0,'time step must be > 0'
        self.delta_x=self.set_min_delta_x()
        self.delta_t=delta_t
        self.delta_t_min=10  # every 10 sec
        self.delta_temp_max=5. # maximum increase of temperature in one step
        self.thickness=self.get_total_thickness()
        self.min_thickness=self.get_minLayer_thickness()
        self.set_initial_temperature()
        # get minimum time step (from boundary condition)
        # get minimum distance
        #
        #x_min=self.create_delta_x()
        #self.t_grenzschicht=self.min_thickness*0.1 ## additional points distance at each boundary layer
        # redifine model finer
        # keypoints without virtual outer points, diffusivity without virtual outer elements 
        # initial_temperatures without virtual nodes
        # layer index field referencing to layer transitiion nodes in virtual node range 
        self.keypoints,self.diffusivity,self.initial_temperatures,self.layer_ix = self.create_discretization()
        self.x = self.keypoints
        self.extend_system()  # create virtual nodes outside
        assert len(self.keypoints)==len(self.diffusivity)+1
        assert len(self.keypoints)==len(self.initial_temperatures)
        self.bc_index()
        self.set_colors()
        self.num_points=len(self.keypoints)
        self.get_system()
        self.set_critical_time()
        
    def set_colors(self):
        colors=['coral','darkgrey','cyan','blue', 'brown', 'burlywood', 'cadetblue']
        #colors=np.arange(10).reshape(2,5).T.ravel()
        ixlist=np.arange(len(colors))
        
        for mname,mobj in Layout.dictofMaterials.items():
            ixlist = np.roll(ixlist,1)
            if mobj.color==0:
                mobj.color=colors[ixlist[0]]
            
        
    
    def get_layer_material_names(self):
        matlist=[]
        for lay in Layout.listofLayers:
            matlist.append(lay.material.name)
        return matlist
    def get_layer_names(self):
        laylist=[]
        for lay in Layout.listofLayers:
            laylist.append(lay.name)
        return laylist
    def get_layer_colors(self):
        colorlist=[]
        for lay in Layout.listofLayers:
            colorlist.append(lay.material.color)
        return colorlist
    
    def bc_index(self):
        '''
        find index in keypoints that is closest to bc position
        '''
        for bc in Layout.listofBC:
            if bc.position < 0:
                bc.position=self.thickness  # total thickness (outer most point)
            dist=np.abs(bc.position-self.keypoints)
            bc.ix=np.argmin(dist)
        
    def set_initial_temperature(self):
        for layer in Layout.listofLayers:
            if layer.initial_temperature is None:
                layer.initial_temperature=self.default_temperature
            
    def get_total_thickness(self):
        th=0
        for layer in Layout.listofLayers:
            th+=layer.thickness
        return th
    
    def get_minLayer_thickness(self):
        th=1e38
        for layer in Layout.listofLayers:
            th=min(th,layer.thickness)
        return th
    def set_min_delta_x(self):
        dx_min=1e38
        for layer in Layout.listofLayers:
            dx_min=min(dx_min,layer.delta_x,layer.thickness/3.)
        return dx_min
       
    
    def extend_system(self):
        '''
        two virtual points at each side are created to consider boundary conditions:
        this allows to use the algorithm for the real outside nodes without changes
            standard: Temperature given outside (top and bottom)
            isothermal: create symmetry to last two real nodes (no heat flux to outside)
            convection: virtual node is ambient, connection is surface transition
        '''
        
        self.keypoints = self._reflect_ends(self.keypoints)
        self.initial_temperatures=self._double_ends(self.initial_temperatures)
        self.diffusivity = self._double_ends(self.diffusivity)
        # if convection replace first and/or last  entry according to film coefficient
        
    def get_system(self):
        '''
        create system matrice/ vector)
    
        '''
        #xp = self.alpha_perdx2
        #### all fixed values except u , and if not temperature dependent!
        #ki=alpha[1:-1]
        #kim1=alpha[:-2]
        #kip1=alpha[2:]

        self.keypoints_dx=self.keypoints[1:]-self.keypoints[:-1]        
        self.alpha_perdx2=self.diffusivity/(self.keypoints_dx*self.keypoints_dx)
        #(ki+kim1) * u[:-2] -( kip1 + ki +ki + kim1 ) * u[1:-1] + (ki+kip1) * u[2:]
        
        
    def set_critical_time(self,limit=0.95):
        
        self.t_critical = 0.5/(self.alpha_perdx2.max())
        self.t_critical_limit = self.t_critical*limit  # go up to 20% to limit


    def _double_ends(self,x):
        return np.r_[x[0],x,x[-1]]

    def _reflect_ends(self,x):
        '''
        negativ on left side, pos. on right side
        '''
        return np.r_[x[0]-(x[1]-x[0]),x,x[-1]+(x[-1]-x[-2])]

        
    def create_discretization(self):
        '''
        from all layers 
        create
        
        discrete points positions 
        associates alpha values between points
        and temperatures at points
        
        '''
        points=[]
        alpha=[]
        temperature=[]
        start=0.
        ix=np.array([0])  # indexfield for layer transitions in tfield (without virtual points)
        for layer in Layout.listofLayers:
            #th_virtual=layer.thickness*layer.material.diffusivity
            #n_points=max(int(layer.thickness/delta_x),2)
            start_index=ix[-1]
            #n_points=layer.n_points
            n_points=max(int(layer.thickness/self.delta_x)+1,3)
            layer.index=np.arange(n_points)+start_index
            ix=np.append(ix,layer.index[-1])
            points_local=np.linspace(0,layer.thickness,num=n_points)
            # consider transition zone (not good, reduces time step!!, removed)
            #if points_local[1]-points_local[0]> self.t_grenzschicht:                
                #points_local=np.insert(points_local,1,self.t_grenzschicht)
                #layer.points_local=np.insert(points_local,-1,points_local[-1]-self.t_grenzschicht)
            #    pass  # creates unregular mesh
            #    layer.points_local=points_local
            #else:
            layer.points_local=points_local
            layer.points_global=start+layer.points_local
            # associate aplha values
            alpha_local=[layer.material.diffusivity]*(len(layer.points_local)-1)
            temperature_local=[layer.initial_temperature]*len(layer.points_local)
            if start == 0.:                
                points.append(layer.points_global)
                temperature.append(temperature_local)
            else:
                points.append(layer.points_global[1:]) # skip 1st node
                temperature[-1][-1]=(temperature[-1][-1]+temperature_local[0])*0.5
                temperature.append(temperature_local[1:])

            alpha.append(alpha_local)
            start=layer.points_global[-1]
            
            
        return np.concatenate(points),np.concatenate(alpha),np.concatenate(temperature),ix
    
    def pprint(self):
        txt='Layout %s \n'%self.name
        for layer in Layout.listofLayers:
            txt += '%s\n'%layer.pprint()
        #txt+='thickness_boundary layer= %f\n'%self.t_grenzschicht
        
        print(self.keypoints)
        print(self.diffusivity)
        
        s=','.join([' %.2f'%x for x in self.keypoints])
        txt+='Keypoints %s\n'%s
        s=','.join([' %.2f'%x for x in self.initial_temperatures])
        txt+='Temperatures %s\n'%s
        txt+='number discretization points %d\n'%self.num_points
        txt+='min element length %f\n'%self.delta_x
        txt+='critical time step: %f\n'%self.t_critical
        return txt

    def compute_temperature_field(self,endtime=None):
        # define end of analysis
        
        

        tfield=np.ndarray((0,self.num_points))
        # if not defined use  max defined time in BC
        if endtime is None:
            endtime=0
            for bc in Layout.listofBC:
                endtime=max(endtime,bc.timepoints[-1])
        
        actual_time=0.0
        tfield=self.initial_temperatures.reshape(1,self.num_points)
        
        delta_t=self.delta_t
        time_list=[0]
        max_step=20000
        nstep=0
        start=tm.time()
        while actual_time<endtime and nstep < max_step:
            
            nstep+=1
            actual_time+=delta_t
            time_list.append(actual_time)
            # last known temperature distribution
            tlast=tfield[-1]
            # new increments
            dudt = self.heat_flow(delta_t,tlast)
            tc=tlast+dudt
            # reflect temperatures
            # to be changed in convection bc
            tc[0]=tc[2]
            tc[-1]=tc[-3]
            # time step control
            #if nstep==19000:
            #    pass
            #    print(tc.max(),tc.min(),np.abs(dudt).max())
            delta_temp = max(np.abs(dudt).max(),1.e-1)
            scale = self.delta_temp_max/delta_temp
            delta_t = min(delta_t*scale,self.t_critical_limit,self.delta_t_min)
            if np.mod(nstep,1000)==0:
                print ('n, time, dtime, d_temp, %d, %f, %f, %f'%(nstep,actual_time,delta_t,delta_temp))
            #delta_t = min(delta_t*scale,self.delta_t_min)
            for bc in Layout.listofBC:
                tc[bc.ix]=bc.get_temperature(actual_time)
            tfield=np.vstack((tfield,tc))
        # do not store extra points from BC (first abd kast colmn)
        t_used=tm.time()-start
        print('time used for solving: %f'%t_used)
        # remove virtual end points
        self.temperature_field= tfield[:,1:-1]
        self.time_points=np.array(time_list)
        return time_list,tfield[:,1:-1]
            
    def heat_flow(self,delta_t,u):
        '''
        Parameters:
            u  array
            len(n) array with temperatures with outermost nodes (they are virtual)
            so bc at real survace are at u[1] and u[-2] of this vector
        returns:
        
            dudt array
            temperature increments at inner nodes (only real ones)
            at outer zero
        '''
        dudt = np.zeros(u.shape)        #Boundary conditions are considered after this routine
        xp = self.alpha_perdx2*delta_t
        dudt[1:-1] = xp[:-1]*(u[:-2]-u[1:-1]) + xp[1:]*(u[2:]-u[1:-1])
        return dudt
                


class Material(object):
    
    def __init__(self,name='Material1',alpha=0,rho=0,cp=0,kh=0,cure={},color=0):
        '''
        alpha    float
            thermal diffusivity (Temperaturleitfaehigkeit)  alpha=kh/(rho*cp)
        rho float
            density
        cp  float
            specific heat capacity
        kh  float
            heat conduction coefficient
        '''
        if alpha==0:
            assert rho > 0, 'rho must be > 0'
            assert cp > 0, 'cp must be > 0'
            assert kh > 0, 'conductivity must be > 0'
            self.diffusivity = kh/(rho*cp)
        else:            
            assert alpha> 0,' Diffusivity must be > 0'
            self.diffusivity = alpha
        self.density = rho
        self.conductivity = kh
        self.heat_capacity = cp
        if name in Layout.dictofMaterials:
            name+='_copy'
        self.name=name
        self.color=color
        Layout.dictofMaterials[name]=self
    def pprint(self):
        txt = 'Material Name: %s \n'%self.name            
        txt += 'Diffusivity %s'%self.diffusivity
        return txt
        
class Layer(object):
    def __init__(self,name='Layer',material=None,thickness=0,initial_temp=None):
        
        self.name=name
        assert material.name in Layout.dictofMaterials,'Material not defined'
        self.material=material
        assert thickness > 0, 'thickness must be > 0'
        self.thickness=thickness
        self.initial_temperature=initial_temp
        self.delta_x= np.sqrt(10* self.material.diffusivity) # min discretization ???
        self.n_points=max(int(self.thickness/self.delta_x)+1,3)
            

        Layout.listofLayers.append(self)
    def update(self,*keyw,**args):
        #myargs=
        for key in args:
            if key =='name':
                self.name=args[key] ##???
            if key == 'thickness':
                self.thickness=args[key]                
            if key == 'material':
                self.material=args[key]
            if key == 'initial_temp':
                self.initial_temperature=args[key]

        self.delta_x= np.sqrt(10* self.material.diffusivity) # min discretization ???
        self.n_points=max(int(self.thickness/self.delta_x),2)
            
    def pprint(self):
        txt='Layer Name:  %s '%self.name        
        txt+=', Material: %s'%Layout.dictofMaterials[self.material.name].name
        txt+=', thickness = %f'%self.thickness        
        txt+=',initial Temperature = %f'%self.initial_temperature
        return txt
    
class Boundary(object):
    def __init__(self,name='BC',timepoints=[],temperatures=[],position=0):
        '''
        define Boundary Condition as time - temperature array
        
        Parameters:
        ----------
        name str
        
        
        position  float
            give position in length in layout (-1: bottom side, 0=top side)
        '''
        self.name=name
        assert len(timepoints) == len(temperatures),'Timepoints and Temperatures must match'
        assert timepoints[0]==0,'Time History has to start at zero'
        self.timepoints=timepoints
        self.temperatures=temperatures
        self.ix=0
        self.bc=np.array([timepoints,temperatures]).T
        if position>=0: # take as x coordinate
            self.position=position
        elif position==-1:#          take at last index (bottom)
            self.position=-1
        else:
            # stop
            assert position==-1,'position of BC can only be -1 or positive, %f'%position            
        Layout.listofBC.append(self)
        
    def pprint(self):
        txt='Boundary %s\n'%self.name
        txt+=self.ndprint(self.bc)
        return txt
    
    def ndprint(self,x):
        txt=''
        for i,(time,temp) in enumerate(x):
            txt+='%d  %.2f  %.2f\n'%(i,time,temp)
        return txt
    def get_temperature(self,time):
        return np.interp(time,self.bc[:,0],self.bc[:,1])
    
    def plot(self):
        pass

class Temperature_field(object):
    def __init__(self,x,time,t_field):
        self.x=x
        self.time=time
        self.temp_field=t_field
    def func_trigger(self,nt=1001):
        '''
         create a regular field with nt timestep
        '''
        tt_field=np.ndarray(shape=(0,nt))
        timet=np.linspace(self.time[0],self.time[-1],num=nt)
        for temp_x in self.temp_field.T:
            tt_x=np.interp(timet,self.time,temp_x).reshape(1,nt)
            tt_field=np.vstack((tt_field,tt_x))
        return timet,tt_field.T
#import dash
#import dash_core_components as dcc
#import dash_html_components as html
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly import tools

import plotly.graph_objs as go
#import plotly.plotly as py
import plotly.figure_factory as ff


def plot2d(time,x, temp_field,boundary_ix=[]):
    
    
    hlines=[]
    trace_lines=[]
    for ix in boundary_ix:
        hline = {       'type': 'line',
                          'x0': x[ix],
                          'y0': time[0],
                          'x1': x[ix],
                          'y1': time[-1],
                          'line': {'color':'black','width':2}
                          }
                          
    hlines.append(hline)
        
    #layout = {'shapes': hlines }
    layout = go.Layout(
            title=go.layout.Title(text='Heat Flow Map',x=0.5),
            xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text='position')),
            yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text='time')),
            )
    layout['shapes']=hlines
    
    
        
    trace_contour=go.Contour(    
        x=x,
        y=time,
        z=temp_field,
        showscale=False,
        contours=dict(
            coloring ='heatmap',
            showlabels = True,
            labelfont = dict(
                family = 'Raleway',
                size = 12,
                color = 'white',                
            )
            
        ),
        line=dict(smoothing=0.85),
        #layout=layout
        #shapes={'type': 'line',
        #        'x0': 20,        
        #        'y0': 0,
        #       'x1': 20,
        #       'y1': 100}
        
    )
    
    
    fig1 = {'data': [trace_contour],
              'layout': layout}
    #plot(fig1,filename='single.html')
    #layout_2=go.Layout()
    plot(fig1,filename='heatmap.html')


def plot_section(belt):
    '''
    plot belt setup
    
    not working
    '''
    layer_nodes=belt.x[belt.layer_ix]
    layer_names=belt.get_layer_names()
    layer_materials=belt.get_layer_material_names()
    layer_colors=belt.get_layer_colors()
    layer_colors.append('black')
    #layer_thickness=layer_nodes[1:]-layer_nodes[:-1]
    layer_center_points = 0.5*(layer_nodes[1:]+layer_nodes[:-1])
    trace = go.Heatmap(
                x=layer_nodes,
                y=['Layer','Material'],
                z = [layer_colors,layer_colors],
                text=[layer_names])
    trace_lines=[]
    for xpos,col in zip(layer_nodes,layer_colors):
        trace =go.Scatter(x=[xpos,xpos],
                          y=[0,1],
                          line={'color':col,'width':2},
                          showlegend=False,
                          mode='lines',
                          fill='tonextx'
                          )
        trace_lines.append(trace)
        
    
    tracea= go.Scatter(
            x=layer_center_points,
            y=[0.5]*len(layer_center_points),
            mode='markers+text',
            text=layer_materials,
            textposition='bottom center'
            )
    trace_lines.append(tracea)
    #x=layer_nodes
    #y=['Layer']
    #z = [layer_colors]
    #z_text=[layer_names]
    #fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Viridis')
    plot(trace_lines,filename='layers.html')

    
def plot2d_subplots(time,x, temp_field,boundary_ix=[],xcuts=[],tcuts=[]):
    
    
    hlines=[]
    trace_lines=[]
    for ix in boundary_ix:
        hline = {       'type': 'line',
                          'x0': x[ix],
                          'y0': time[0],
                          'x1': x[ix],
                          'y1': time[-1],
                          'line': {'color':'black','width':2}
                          }
        trace =go.Scatter(x=[x[ix],x[ix]],
                          y=[time[0],time[-1]],
                          line={'color':'black','width':2},
                          showlegend=False,
                          mode='lines'
                          
                          )
        hlines.append(hline)
        trace_lines.append(trace)
        
    layout = {'shapes': hlines }

    
    
        
    trace_contour=go.Contour(
        x=x,
        y=time,
        z=temp_field,
        showscale=False,
        contours=dict(
            coloring ='heatmap',
            showlabels = True,
            labelfont = dict(
                family = 'Raleway',
                size = 12,
                color = 'white',
            )
        ),
        line=dict(smoothing=0.85),
        #layout=layout
        #shapes={'type': 'line',
        #        'x0': 20,        
        #        'y0': 0,
        #       'x1': 20,
        #       'y1': 100}
        
    )
    traces_xcut=[]
    if not xcuts:
        xcuts=[x[-1]/2.]
    for xcut in xcuts:          
        dx=x-xcut
        ix_cut=np.argmin(abs(dx))
        trace_heat=go.Scatter(        
            x = temp_field[:,ix_cut],
            y = time,
            showlegend=True,
            name='%.4f'%x[ix_cut]
            )
        traces_xcut.append(trace_heat)


    traces_tcut=[]
    if not tcuts:
        tcuts=[time[-1]/2.]
    for tcut in tcuts:          
        dt=time-tcut
        ix_cut=np.argmin(abs(dt))

    
        #tcut=int(temp_field.shape[0]/2)
        trace_time=go.Scatter(        
            y = temp_field[ix_cut,:],
            x = x,
            showlegend=True,
            name='%.1f s'%time[ix_cut]
            )
        traces_tcut.append(trace_time)
    
    fig1 = {'data': [trace_contour],
              'layout': layout}
    #plot(fig1,filename='single.html')
    #layout_2=go.Layout()
    fig = tools.make_subplots(rows=2,cols=4,subplot_titles=('Temperature History','Temperature','Curing','Curing History'))
    #
    fig.append_trace(trace_contour,2,2)
    fig.append_trace(trace_contour,2,3)
    fig.append_trace(trace_heat,2,1)
    fig.append_trace(trace_heat,2,4)
    fig.append_trace(trace_time,1,2)
    fig.append_trace(trace_time,1,3)
    for t in trace_lines:
        fig.append_trace(t,2,2)
        fig.append_trace(t,2,3)
        fig.append_trace(t,1,2)
        fig.append_trace(t,1,3)
    
    # wirkt im ersten subplot (1,1)
    fig['layout'].update(showlegend=False)
    #fig['layout'].update(
    #       annotations=[
    #               dict(x=2,y=5,xref='x',yref='y',
    #                    text='text', ax=100, ay=-40)
    #               ])
    
    
# =============================================================================
#     layout_sub=go.Layout(yaxis=dict(domain=[0,0.3]),
#                          yaxis2=dict(domain=[0.33,0.80]),
#                          yaxis3=dict(domain=[0.85,1]))
#     fig.layout.update(layout_sub)
#     
# =============================================================================
    plot(fig,filename='double.html')
    
    if 0:
        fig_all=go.Figure()
        fig_all.add_traces([trace_contour,trace_contour])
        
        fig_all.layout.update(layout)
        # nur ein Plot!
        plot(fig_all,filename='double_all.html')
        
    #fig.add_trace()
    '''
        
        trace=go.Trace(
        x = [x[ix],x[ix]],
        y = [time[0],time[-1]],
        type="scatter",
        mode = "line")
        fig.add_trace(trace)
        #fig.add_trace(trace,1,2)
        
       ''' 
def plot_fix_location(x,time,field,x_list):
    '''
    plot tempertur vs. time at given locations
    '''
    #from scipy import interpolate
    #nt=len(time)
    #xx,yy = np.meshgrid(x,time)
    #f = interpolate.interp2d(x,time,field)
    traces=[]
    for xcut in x_list:
        vixmin = np.argsort(abs(xcut-x))
        #ixmin=vixmin[0]
#        if x[ixmin]-xcut < 0:  # ixmin is left of xcut
#            ixleft=ixmin
#            ixright=vixmin[1]
#        else:                  # ixmin is right of xcut or on cut
#            ixleft=vixmin[1]
#            ixright=ixmin
        dx = x[vixmin[1]]-x[vixmin[0]]        
        xinterpfactor=(xcut-x[vixmin[0]])/dx
        x_temp=field[:,vixmin[0]]+xinterpfactor*(field[:,vixmin[1]]-field[:,vixmin[0]])
        for xt in [x_temp]:#[field[:,vixmin[0]], x_temp, field[:,vixmin[1]]]:
            trace=go.Scatter(        
                x = time,
                y = xt,
                showlegend=True,
                name='x=%.5g'%xcut
                )
            
            traces.append(trace)
        
    fig={'data': traces}#,  'layout': layout}
    plot(fig,filename='heat_vs_time.html')
    
def plot_fix_time(x,ltime,field,time_list):
    '''
    plot tempertur vs. time at given locations
    '''
    #from scipy import interpolate
    #nt=len(time)
    #xx,yy = np.meshgrid(x,time)
    #f = interpolate.interp2d(x,time,field)
    traces=[]
    atime=np.array(ltime)
    for tfix in time_list:
        vixmin = np.argsort(abs(tfix-atime))
        #ixmin=vixmin[0]
#        if x[ixmin]-xcut < 0:  # ixmin is left of xcut
#            ixleft=ixmin
#            ixright=vixmin[1]
#        else:                  # ixmin is right of xcut or on cut
#            ixleft=vixmin[1]
#            ixright=ixmin
        dt = atime[vixmin[1]]-atime[vixmin[0]]        
        xinterpfactor=(tfix-atime[vixmin[0]])/dt
        t_temp=field[vixmin[0],:]+xinterpfactor*(field[vixmin[1],:]-field[vixmin[0],:])
        for xt in [t_temp]:#[field[:,vixmin[0]], x_temp, field[:,vixmin[1]]]:
            trace=go.Scatter(        
                x = x,
                y = xt,
                showlegend=True,
                name='%.5g s'%tfix
                )
            
            traces.append(trace)
        
    fig={'data': traces}#,  'layout': layout}
    plot(fig,filename='heat_vs_location.html')
    
def plot_mpl(time,x, temp_field):
    #import matplotlib.pyplot as plt
    xx,yy = np.meshgrid(x,time)
    h=plt.contourf(xx,yy,temp_field)
    plt.show()
    

        
            
if __name__ =='__main__':
    M1=Material('m1',alpha=0.15)
    M2=Material('m2',alpha=0.15)
    print(M1.pprint())
    L1=Layer('1.layer',material=M1,thickness=10,initial_temp=50)
    L2=Layer('2.layer',material=M2,thickness=10,initial_temp=50)
    L1.n_points=3
    L2.n_points=11
    print(L1.pprint())
    #L1.update(thickness=20)
    print(L1.pprint())
    top=Boundary('top',timepoints=[0,10,100,1000],temperatures=[80,120,160,160],position=0)
    bot=Boundary('bot',timepoints=[0,10,100],temperatures=[80,120,160],position=-1)
    print(top.pprint())
    print(bot.pprint())

    belt=Layout(default_temperature=90)
    print(belt.pprint())
    
    time,temp=belt.compute_temperature_field()
    print (belt.temperature_field)
    #plot_fix_location(belt.x,time,temp,[1,2,3,4])
    #plot_fix_time(belt.x,time,temp,[0,20,33,44])
    plot_section(belt)
    plot_mpl(time,belt.x,temp)
    #plot2d(time,belt.x,temp,belt.layer_ix)
    '''
    top.temperatures=[80,120,180,140]
    time,temp=belt.compute_temperature_field()
    print (belt.temperature_field)
    plot_mpl(time,belt.x,temp)
    np.savetxt('time.csv',time,delimiter=',')
    np.savetxt('xpos.csv',belt.x,delimiter=',')
    header = ','.join(map(str,range(temp.shape[1])))
    np.savetxt('heat.csv',temp,delimiter=', ',header = header)
    print(belt.get_layer_material_names())
    #layer_nodes=[],layer_names=[],layer_materials=[],layer_colors=[]):
    #plot_section(belt.x[belt.layer_ix],belt.get_layer_names(),belt.get_layer_material_names(),belt.get_layer_colors())
    
    '''
    
"""
'''
Layer Class for saving the material constants as an object.

    Variables:
        rho:        Float
        cp:         Float
        k:          Float
        thickness:  Float
        color:      String of Hexadecimal Color (#000000)
        gummi:      Empty Dict or Dict of Rubber-Constants
'''
class Layer():
    '''
        Initialize Layer for all Variables of Layer()
    '''
    def __init__(self, rho, cp, k, thickness, color, gummi):
        self.rho = float(rho)
        self.cp = float(cp)
        self.k = float(k)
        self.thickness = float(thickness)
        self.color = color
        self.gummi = gummi

    #Return Function for testing.
    def __str__(self):
        return 'Rho: %.2f, Cp: %.2f, K: %.2f, thickness: %.2f' %(self.rho, self.cp, self.k, self.thickness)

'''
Temperature Distribution Class for saving the Distribution and acess all its explicit and implicit points

    Variables
        data:               Dict of Dict with ['timestep'] and ['temperature'] field
        lookupGradient:     Dict with ['timestep'] and ['gradient'] field
        lookupTemperature:  Dict with ['timestep'] and ['temperature'] field
'''
class Tempdist():
    '''
    # Uncaught if starting and finishing point of the two temperature
    # distributions differ from each other.
    '''

    '''
        Initialize Distribution with all Datapoints and empty lookup Fields
    '''
    def __init__(self, data):
        self.data = data
        self.lookupGradient = {}
        self.lookupTemperature = {}

    #Return Function for testing.
    def __str__(self):
        return 'Points: %i, other info...' %(len(self.data))

    '''@returns first Timestep'''
    def getFirstTimestep(self):
        return self.data[0]['timestep']

    '''@returns first Temperature'''
    def getFirstTemperature(self):
        return self.data[0]['temperature']

    '''@returns last Timestep'''
    def getLastTimestep(self):
        return self.data[-1]['timestep']

    '''@returns last Temperature'''
    def getLastTemperature(self):
        return self.data[-1]['temperature']

    '''@returns total Time'''
    def getTotalTime(self):
        return self.getLastTimestep() - self.getFirstTimestep()

    def getTemperature(self, time):
        if time in self.lookupTemperature.keys():
            return self.lookupTemperature[time]
        if time < self.getFirstTimestep():
            self.lookupTemperature = {time : self.getFirstTemperature()}
            return self.getFirstTemperature()
        elif time > self.getLastTimestep():
            self.lookupTemperature = {time : self.getLastTemperature()}
            return self.getLastTemperature()
        else:
            #Check if timestep is in list
            if time in [x['timestep'] for x in self.data]:
                index = [x['timestep'] for x in self.data].index(time)
                return self.data[index]['temperature']
            #Find time intervall
            for i in range(len(self.data)-1):
                if self.data[i]['timestep'] < time < self.data[i+1]['timestep']:
                    intervall = i
                    break
            #Get approximate Temprerature
            #   Gradient
            gradient = float(self.data[intervall+1]['temperature'] - self.data[intervall]['temperature']) / float(self.data[intervall+1]['timestep'] - self.data[intervall]['timestep'])
            #   Point on line
            point = self.data[intervall]['temperature'] + gradient * (time - self.data[intervall]['timestep'])
            self.lookupTemperature = {time : point}
            return point

    def getGradient(self, time):
        if time in self.lookupGradient.keys():
            return self.lookupGradient[time]
        if  time < self.getFirstTimestep():
            self.lookupGradient = {time : 0}
            return 0
        elif time > self.getLastTimestep():
            self.lookupGradient = {time : 0}
            return 0
        else:
            #Find time intervall
            for i in range(len(self.data)-1):
                if self.data[i]['timestep'] <= time <= self.data[i+1]['timestep']:
                    intervall = i
                    break
            #Get approximate Temprerature
            #   Gradient
            gradient = float(self.data[intervall+1]['temperature'] - self.data[intervall]['temperature']) / float(self.data[intervall+1]['timestep'] - self.data[intervall]['timestep'])
            self.lookupGradient = {time : gradient}
            return gradient

def plotSolution(solution, appendix):
    for i in range(0, len(t), int(np.ceil(len(t)/10))) + [len(t)-1]:
        plt.plot(x, solution[i], label='t={0:1.2f}'.format(t[i]))

    # put legend outside the figure
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('X position')
    plt.ylabel('Temperature')
    plt.grid(True)

    # adjust figure edges so the legend is in the figure
    plt.subplots_adjust(top=0.89, right=0.77)
    plt.savefig(os.path.dirname(os.path.abspath(__file__)) + '\\pde-transient-heat-' + appendix + '.png')

#    # animated solution. We will use imagemagick for this
#
#    # we save each frame as an image, and use the imagemagick convert command to
#    # make an animated gif
#    for i in range(len(t)):
#        plt.clf()
#        plt.plot(x, solution[i])
#        plt.xlabel('X')
#        plt.ylabel('T(X)')
#        plt.grid(True)
#        plt.title('t = {0}'.format(t[i]))
#        plt.savefig('___t{0:03d}.png'.format(i))
#
#    import subprocess
#    subprocess.call('magick convert -quality 100 ___t*.png images/transient_heat.gif', shell=True)
#    subprocess.call('del ___t*.png', shell=True)

def plotContour(solution, appendix):
    #print solution
    fontsize = 18

    if appendix > 0:
        a = [x[i] for i in range(len(x)) if i in sum(CureRange,[])]
        sol = solution[:]
        for i in range(len(sol)):
            for j in reversed(range(len(sol[i]))):
                if j not in sum(CureRange,[]):
                    del sol[i][j]
    else:
        a = x[:]
        sol = solution[:]


    X, Y = np.meshgrid(a, t)
    plt.figure()

    CS = plt.contour(X, Y, sol, colors='k', alpha=1, linewidths=1.5, antialiased=True)
    plt.clabel(CS, inline=True, fontsize=fontsize, fmt='%1.f')
    #plt.xlabel('X position', fontsize=fontsize)
    #plt.ylabel('Time', fontsize=fontsize)

    ax1 = plt.gca()
    summe = 0
    xcoords = []
    for layer in layers:
        if layer.gummi or appendix == 0:
            ax1.add_patch(patches.Rectangle((summe,0),layer.thickness,Time, alpha=0.5, facecolor=layer.color, edgecolor="none"))
        summe += layer.thickness
        xcoords.append(summe)

    if appendix == 0:
        for xc in xcoords[:-1]:
            plt.axvline(x=xc, color='#000000', alpha=1)

    ax1.tick_params(axis='both', which='major', labelsize=fontsize)
    #ax1.set_xticks(ax1.get_xticks()[::2])
    #ax1.set_yticks(ax1.get_yticks()[::2])
    ax1.grid(linestyle='solid', alpha=0.3)
    [i.set_linewidth(2) for i in ax1.spines.values()]

    plt.savefig(os.path.dirname(os.path.abspath(__file__)) + '\\pde-transient-heat-contour-' + str(appendix) + '.png', transparent=False, bbox_inches='tight')

def odefunc2(currentTime, u):
    dudt = np.zeros(x.shape)        #Boundary conditions!
    u[0] = outer.getTemperature(currentTime)
    u[-1] = inner.getTemperature(currentTime)

    # now for the internal nodes
    for i in range(1, x.shape[0]-1):
        dudt[i] = 1 / (2 * np.power(x[i]-x[i-1],2) * rho[i] * cp[i]) * ((k[i]+k[i-1]) * u[i-1] - (k[i+1] + 2 * k[i] + k[i-1]) * u[i] + (k[i]+k[i+1]) * u[i+1])
        #dudt[i] = 1/(2 * np.power(x[i]-x[i-1],2)) * ((alpha[i] + alpha[i-1]) * u[i-1] - (alpha[i+1] + 2 * alpha[i] + alpha[i-1]) * u[i] + (alpha[i] + alpha[i+1]) * u[i+1])

    return dudt

def odefunc2a(currentTime, u):
    dudt = np.zeros(x.shape)        #Boundary conditions!
    u[0] = outer.getTemperature(currentTime)
    u[-1] = inner.getTemperature(currentTime)

    # now for the internal nodes
    #for i in range(1, x.shape[0]-1):
    #    dudt[i] = 1 / (2 * np.power(x[i]-x[i-1],2) * rho[i] * cp[i]) * ((k[i]+k[i-1]) * u[i-1] - (k[i+1] + 2 * k[i] + k[i-1]) * u[i] + (k[i]+k[i+1]) * u[i+1])
        #dudt[i] = 1/(2 * np.power(x[i]-x[i-1],2)) * ((alpha[i] + alpha[i-1]) * u[i-1] - (alpha[i+1] + 2 * alpha[i] + alpha[i-1]) * u[i] + (alpha[i] + alpha[i+1]) * u[i+1])
    xd  = (x[1:-1]-x[:-2])
    xp = 1./(2*xd*xd *rho[1:-1]*cp[1:-1])
    ki=k[1:-1]
    kim1=k[:-2]
    kip1=k[2:]
    #xk = (k[1:-1]+k[:-2]) * u[:-2] - (k[2:] + 2*k[1:-1] + k[:-2]) *u[1:-1] + (k[1:-1]+k[2:]) * u[2:]
    xk = (ki+kim1) * u[:-2] -( kip1 + ki +ki + kim1 ) * u[1:-1] + (ki+kip1) * u[2:]
    dudt[1:-1] = xp * xk
    return dudt

'''
handle = open(sys.argv[1], 'r')
txt = handle.read()
txt=txt.replace(u'\x00','')
data = yaml.safe_load(txt)
handle.close()
'''
# layer1

layers = []
cureobjects = []
'''
for layer in data['layers']:
    layers.append(Layer(layer['rho'], layer['cp'], layer['k'], layer['thickness'], layer['color'], layer['gummi']))
    #Jan's Vulkanisierungsansatz

    if layer['gummi']:
        cureobjects.append(vc.vulcameter(
            a=layer['gummi']['a'],
            b=layer['gummi']['b'],
            c=layer['gummi']['c'],
            tau=layer['gummi']['tau'],
            tm=layer['gummi']['tm'],
            ti=layer['gummi']['ti'],
            ml=layer['gummi']['ml'],
            mh=layer['gummi']['mh']))
'''        

layers.append(Layer(1.17e3,1600.,0.35,0.5e-3,'r',False))# bottom
layers.append(Layer(1.17e3,1600.,0.35,3.5e-3,'r',False))
'''
layers.append(Layer(1.28,1600.,0.35,0.7,'b',False))
layers.append(Layer(1.20,1600.,0.15,1.4,'g',False))
#layers.append(Layer(1.28,1600.,0.35,0.17,'b',False))

#layers.append(Layer(1.28,1600.,0.35,0.35,'b',False))
#layers.append(Layer(1.20,1600.,0.15,0.7,'g',False))
#layers.append(Layer(1.28,1600.,0.35,0.17,'b',False))

layers.append(Layer(1.28,1600.,0.35,0.7,'b',False))
layers.append(Layer(1.20,1600.,0.15,1.4,'g',False))
#layers.append(Layer(1.28,1600.,0.35,0.17,'b',False))
'''
# original fabric
for i in range(4):
    layers.append(Layer(1.28e3,1600.,0.35,0.175e-3,'b',False))
    layers.append(Layer(1.20e3,1600.,0.15,0.7e-3,'g',False))
    layers.append(Layer(1.28e3,1600.,0.35,0.175e-3,'b',False))

layers.append(Layer(1.17e3,1600.,0.35,5.5e-3,'r',False))
layers.append(Layer(1.17e3,1600.,0.35,0.5e-3,'r',False))


        #data:               Dict of Dict with ['timestep'] and ['temperature'] field
tempdata=[{'timestep':0,'temperature':20},
          {'timestep':120,'temperature':162},
          {'timestep':1200,'temperature':162}]


#inner = Tempdist(data['tempdist_inner'])
#outer = Tempdist(data['tempdist_outer'])
inner = Tempdist(tempdata)
outer = Tempdist(tempdata)


Length = sum([x.thickness for x in layers])           #Length of the construct
Time = inner.getTotalTime()                           #Time in seconds

n_x = np.empty(len(layers),dtype=np.int64)
n_x.fill(0)

stream = open(os.path.dirname(os.path.abspath(__file__)) + '\\discretization.dat', 'r')
discretization = yaml.load(stream)
stream.close()

dt = discretization['dt']
dx = [discretization['dx'] for i in range(len(layers))]
for i in range(len(layers)):
    n_x[i] = np.ceil(layers[i].thickness / dx[i])

x = np.linspace(0.0, Length, int(np.ceil(Length/dx[0])))
t = np.linspace(0.0, Time, int(np.ceil(Time / dt)))   #Discretization in time
print('Stuetzstellen %d '%len(x))

rho = np.array([])
cp  = np.array([])
k   = np.array([])
alpha = np.array([])
CureRange = []
summ=0
for layer in range(len(layers)):
    rho = np.append(rho, np.ones(n_x[layer]) * layers[layer].rho)
    cp = np.append(cp, np.ones(n_x[layer]) * layers[layer].cp)
    k = np.append(k, np.ones(n_x[layer]) * layers[layer].k)
    alpha = np.append(alpha, np.ones(n_x[layer]) * layers[layer].k / (layers[layer].rho * layers[layer].cp ))
    if layers[layer].gummi:
        CureRange.append([i for i,val in enumerate(x) if summ-1e-10<=val<=summ+layers[layer].thickness+1e-10])
    summ+=layers[layer].thickness
#k = np.append(k, layers[-1].k)
alpha = np.append(alpha, layers[-1].k / (layers[-1].rho * layers[-1].cp))

#initTemp = data['initial_temp'] * np.ones(x.shape)        # initial temperature
initTemp = 20. * np.ones_like(x)        # initial temperature
#Jans vulkanisierungs Berechnung
#   Initial Cure to zero
Alpha = [[1e-36] for i in range(x.size)]
Moment = [[0] for i in range(x.size)]
tmflag = [0 for i in range(x.size)]


#import time
#summ = 0.
import time
timer = time.time()
start = timer

solution = initTemp
solution = []
helpFunction = ode(odefunc2a).set_integrator('lsoda')
helpFunction.set_initial_value(initTemp)
while helpFunction.successful() and helpFunction.t + 1e-10 <= Time:
    currentTime = helpFunction.t + dt
    #start = time.time()
    solution.append(helpFunction.integrate(currentTime))
    #solution=np.vstack((solution,helpFunction.integrate(currentTime)))
    solution[-1][0] = outer.getTemperature(currentTime)
    solution[-1][-1] = inner.getTemperature(currentTime)
    #end = time.time()
    #summ += (end-start)
    #if (end-start) > 0.7: print currentTime, ': ', (end-start), '<br>'
    now=time.time()
    if now-timer > 10:
        print ('time %.1f, dt %f, steps %d'%(currentTime,dt,len(solution)))
        print ('ready at %.2f percent'%(currentTime/Time*100))
        timer=now
        

    #Calculate Cure
    for cureobject in range(len(CureRange)):
        for i in CureRange[cureobject]:
            #Calculate Cure
            flag_new, dAlpha, alpha_new, moment_new = cureobjects[cureobject].cure_increment(currentTime, dt, solution[-1][i] + 273.15, Alpha[i][-1], tmflag[i], Moment[i][-1])
            Alpha[i].append(alpha_new)
            Moment[i].append(moment_new)
            tmflag[i] = flag_new
#print 'Total: ', summ, '<br>'
print ('time used: %f'%(time.time()-start))

for i in [a for a in range(len(solution[0])) if a not in sum(CureRange,[])]:
    Alpha[i] = list(np.zeros(len(solution)))
    Moment[i] = list(np.zeros(len(solution)))

Alpha = [list(i) for i in zip(*Alpha)]
for i in range(len(Alpha)):
    for j in range(len(Alpha[i])):
        Alpha[i][j] = 0.0 if Alpha[i][j] < 1e-34 else Alpha[i][j] * 100.0
Moment = [list(i) for i in zip(*Moment)]

if len(Alpha) > len(solution):
    Alpha = Alpha[1:]
    Moment = Moment[1:]
#plotSolution(solution,0)
plotContour(solution, 0)

#plotContour(Alpha, 1)
#plotContour(Moment, 2)

'''
# Return Results
if len(solution) <= 100:
    NumberOfResults = len(solution)
else:
    NumberOfResults = 100

s = ''
for j in range(0, len(solution), int(np.ceil(len(solution)/NumberOfResults)))+[len(solution)-1]:
    s+= str(t[j]) + ':'
    for i in range(len(solution[j])):
        s+= str(x[i]) +  ',' +  str(solution[j][i]) + ';'
    s = s[:-1] + ':'

s = s[:-1] + '/'
for j in range(0, len(solution), int(np.ceil(len(solution)/NumberOfResults)))+[len(solution)-1]:
    s+= str(t[j]) + ':'
    for i in range(len(Alpha[0])):
        s+= str(x[sum(CureRange,[])[i]]) +  ',' +  str(Alpha[j][i]) + ';'
    s = s[:-1] + ':'

s = s[:-1] + '/'
for j in range(0, len(solution), int(np.ceil(len(solution)/NumberOfResults)))+[len(solution)-1]:
    s+= str(t[j]) + ':'
    for i in range(len(Moment[0])):
        s+= str(x[sum(CureRange,[])[i]]) +  ',' +  str(Moment[j][i]) + ';'
    s = s[:-1] + ':'

print (s[:-1])
#plotSolution(solution,'s2')
'''
sa=np.vstack(initTemp,np.array(solution))
print(sa[0::12,[100,1220,1640,2740]])

"""