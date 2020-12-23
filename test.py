import pandas as pd
import numpy as np
from bokeh.plotting import show, output_file
from bokeh.layouts import column
from pandas import ExcelWriter
import statistics as st
from bokeh.plotting import figure, show, output_file
from bokeh.layouts import layout, row,column
from bokeh.models import Band,LinearAxis, Range1d,DataRange1d,Span,HoverTool,ColumnDataSource,CrosshairTool,BoxAnnotation
from bokeh.models import CheckboxGroup, CustomJS
from bokeh.palettes import Viridis3,RdYlBu3,Category20b_20
from bokeh.models.glyphs import Segment

def inx(lst):
    xx=[]
    for i,a in enumerate(lst):
        if a==0:
            xx.append(i)
    return(xx)        

def date(x):
    x=str(x)
    y=x[:4]  
    m=x[4:6]
    d=x[6:]  
    if m[0]=='0' :
        m=m[1]
    if d[0]=='0':
        d=d[1]
    dt=y+'_'+m+'_'+d  
    return dt
    
def tadil(i,df1,ln,u):
        if i=='معيار':
            path="D:/mohandes/client/"+ i+" -ت.xls"
        else:    
            path="D:/mohandes/client/"+ i+"-ت.xls"
        df2=pd.read_excel(path)
        df2['date']=df2['date'].apply(date)
        df3=u[u.symbol==i]
        #df2=pd.concat([df2,df3])
        
        if i=='بسويچ':
            df2=df2.drop(index=df2[df2.date==13981130].index)
        
        df2=df2[-250:]
        l=len(df2)
        
        opz=inx(list(df2.open))
        opn=list(df2.open)
        for i in opz:
            opn[i]=df2.iloc[i-1]['final']
            df2['open']=opn
        
        df2['perB']=['']*(l-ln)+list(df1.perB)    
        df2['perC']=['']*(l-ln)+list(df1.perC)
        df2['power']=['']*(l-ln)+list(df1.power)
        
        
        return df2

u=pd.read_excel("D:/mohandes/sahm/update.xlsx",sheet_name="Sheet2")
nan= []
for i in range(len(u)):
    if str(u.iloc[i][-1])=="nan" or str(u.iloc[i][-2])=="nan" or str(u.iloc[i][-3])=="nan" :
        nan.append(i)
u=u.drop(index=nan)
u1=u.loc[:,{'symbol','date','perB','perC','power'}]
u=u.drop(columns={'name','close_percent','final_percent','perB','perC','power'})
today=u.iloc[0]['date']
#namad=pd.read_excel("D:/mohandes/New folder (2)/namad.xlsx")
n=list(u.symbol)
i=n[0]
i='بتك'
i='تابا'



for j in range(1):
    df=pd.read_excel("D:/mohandes/power/"+i+".xlsx")
    ln=len(df)
    df=tadil(i,df,ln,u)


    l=len(df)
    p=df[df['power']!='']
    lp=len(p)
    p['valB']=p['value']*p['perB']
    p['valC']=p['value']*p['perC']
    power5=p['valB'].rolling(5).sum()/p['valC'].rolling(5).sum()
    power10=p['valB'].rolling(10).sum()/p['valC'].rolling(10).sum()
    power15=p['valB'].rolling(15).sum()/p['valC'].rolling(15).sum()
    df['power5']=['']*(l-lp)+list(power5)
    df['power10']=['']*(l-lp)+list(power10)
    df['power15']=['']*(l-lp)+list(power15)
    df["SMA"+str(20)]=df['close'].rolling(20).mean()
    df['EMA'+str(20)] = pd.Series.ewm(df['close'], span=20,min_periods=20).mean()
    d20=pd.Series.ewm(df['EMA20'], span=20,min_periods=20).mean()
    t20=pd.Series.ewm(d20, span=20,min_periods=20).mean()
    df['DEMA20']=2*df['EMA20']-d20
    df['TEMA20']=3*df['EMA20']-3*d20+t20
    df['EMA'+str(50)] = pd.Series.ewm(df['close'], span=50,min_periods=50).mean()
    d50=pd.Series.ewm(df['EMA50'], span=50,min_periods=50).mean()
    t50=pd.Series.ewm(d50, span=50,min_periods=50).mean()
    df['DEMA50']=2*df['EMA50']-d50
    df['TEMA50']=3*df['EMA50']-3*d50+t50
    macd=pd.Series.ewm(df['close'], span=12,min_periods=12).mean()-pd.Series.ewm(df['close'], span=26,min_periods=26).mean()
    signal= pd.Series.ewm(macd, span=9,min_periods=9).mean()
    hist=macd-signal 
    df['MACD']=macd
    df['Signal']=signal
    df['Hist']=hist
    mb=df['close'].rolling(20).mean()
    std=df['close'].rolling(20).std()
    df['UB']=mb+2*std
    df['LB']=mb-2*std
    diff = df['close'].diff(1)        
    up_chg = 0 * diff
    down_chg = 0 * diff
    up_chg[diff > 0] = diff[ diff>0 ]
    down_chg[diff < 0] = diff[ diff < 0 ]
    up_chg_avg   = up_chg.ewm(com=14-1 , min_periods=14).mean()
    down_chg_avg = down_chg.ewm(com=14-1 , min_periods=14).mean() 
    rs = abs(up_chg_avg/down_chg_avg)
    rsi = 100 - 100/(1+rs)
    df['RSI'+str(14)]=rsi
    tr1=list(df['high']-df['low'])[1:]
    tr2=[abs(i-j) for i,j in zip(list(df['high'][1:]),list(df['close'][:-1]))]  
    tr3=[abs(i-j) for i,j in zip(list(df['low'][1:]),list(df['close'][:-1]))]  
    tr=[np.nan]+[max(i,j,k) for i,j,k in zip(tr1,tr2,tr3)] 

    upmove=[i-j for i,j in zip(list(df['high'][1:]),list(df['high'][:-1]))] 
    downmove=[i-j for i,j in zip(list(df['low'][:-1]),list(df['low'][1:]))] 
    DMp=[np.nan]
    DMn=[np.nan]
    for i,j in zip(upmove,downmove):
        if i>j and i>0:
            DMp.append(i)
        else:
            DMp.append(0)
        if j>i and j>0:
            DMn.append(j)
        else:
            DMn.append(0)    
    d={'tr':tr,'DMp':DMp,'DMn':DMn}  
    dx= pd.DataFrame(data=d)  
    atr=pd.Series.ewm(dx['tr'],alpha=1/14,min_periods=14).mean()
    sDMp=pd.Series.ewm(dx['DMp'], alpha=1/14,min_periods=14).mean()
    sDMn=pd.Series.ewm(dx['DMn'], alpha=1/14,min_periods=14).mean()
    DIp=sDMp/atr*100
    DIn=sDMn/atr*100
    Dx=pd.Series.ewm(abs((DIp-DIn)/(DIp+DIn))*100 ,alpha=1/14,min_periods=14).mean()
    df['DIp']=list(DIp)
    df['DIn']=list(DIn)
    df['ADX']=list(Dx)
    
    length = l
    high = list(df['high'])
    low = list(df['low'])
    close = list(df['close'])
    psar = close[0:len(close)]
    psarbull = [None] * length
    psarbear = [None] * length
    bull = True
    iaf = 0.02 
    maxaf = 0.2
    af = iaf
    hp = high[0]
    lp = low[0]
    for i in range(2,length):
        if bull:
            psar[i] = psar[i - 1] + af * (hp - psar[i - 1])
        else:
            psar[i] = psar[i - 1] + af * (lp - psar[i - 1])
        reverse = False
        if bull:
            if low[i] < psar[i]:
                bull = False
                reverse = True
                psar[i] = hp
                lp = low[i]
                af = iaf
        else:
            if high[i] > psar[i]:
                bull = True
                reverse = True
                psar[i] = lp
                hp = high[i]
                af = iaf
        if not reverse:
            if bull:
                if high[i] > hp:
                    hp = high[i]
                    af = min(af + iaf, maxaf)
                if low[i - 1] < psar[i]:
                    psar[i] = low[i - 1]
                if low[i - 2] < psar[i]:
                    psar[i] = low[i - 2]
            else:
                if low[i] < lp:
                    lp = low[i]
                    af = min(af + iaf, maxaf)
                if high[i - 1] > psar[i]:
                    psar[i] = high[i - 1]
                if high[i - 2] > psar[i]:
                    psar[i] = high[i - 2]
        if bull:
            psarbull[i] = psar[i]
        else:
            psarbear[i] = psar[i]
    df['PSAR']=psar
    df['PSAR_bear']=psarbear
    df['PSAR_bull']=psarbull   
    
    shif=[str(i) for i in range(1,27)]
    df2=pd.DataFrame(data={'date':shif})
    df=pd.concat([df,df2])
    
    
    high_9 = df['high'].rolling(window= 9).max()
    low_9 = df['low'].rolling(window= 9).min()
    df['tenkan_sen'] = (high_9 + low_9) /2

    high_26 = df['high'].rolling(window= 26).max()
    low_26 = df['low'].rolling(window= 26).min()
    df['kijun_sen'] = (high_26 + low_26) /2


    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)

    high_52 = df['high'].rolling(window= 52).max()
    low_52 = df['low'].rolling(window= 52).min()
    df['senkou_span_b'] = ((high_52 + low_52) /2).shift(26)
    df['chikou_span'] = df['close'].shift(-22)
    
    df=df[-130:-6]
    
    
    
    a=max(df['UB'][df['UB'].notna()])*1.05
    b=max(df['volume'][df['volume'].notna()])*3

    powr=df['power'] !=''
    powr5=df['power5'] !=''
    #c=1.1*max(df['power5'][-sum(powr)+4:])
    #c=1.4*max(df['power'][-sum(powr):].iloc[-30:])
    c=min(1.05*max(df['power'][powr][df.power[powr].notna()].iloc[-20:]),
          1.1*max(df['power5'][powr5][df['power5'][powr5].notna()]))
    
    inc = df['close'] >= df['open']
    dec = df['open'] > df['close']
    
    source=ColumnDataSource(data=df)
            
    w = 0.5

    TOOLS1 = "pan,wheel_zoom,box_zoom,reset,save,crosshair"
    TOOLS2="pan,wheel_zoom,reset"
           
    p = figure(x_range=list(df['date']),tools=TOOLS1, plot_width=1420,plot_height=600,
           title = df.iloc[0]['symbol'],y_range=(0,a),)
    p.yaxis.axis_label="Price"
    p.title.text_font_size="20px"
    p.title.align="center"

    p.extra_y_ranges = {"vol": Range1d(start=0, end=b)}
    p.xaxis.major_label_orientation = np.pi/4 
    p.yaxis.axis_label_text_font_size="15px"  
    p.xaxis.major_label_text_font_size="7px"
    p.add_layout(LinearAxis(y_range_name="vol",axis_label="Volume",axis_label_text_font_size="15px"), 'right')

    p.segment(df['date'], df['low'],df['date'], df['high'], color="black")
    p.vbar(df['date'][inc],w, df['open'][inc], df['close'][inc], fill_color="#D5E1DD", line_color="black",legend="Ascending")
    p.vbar(df['date'][dec],w, df['open'][dec], df['close'][dec], fill_color="#F2583E", line_color="black",legend="Descending")

    p.vbar(df['date'][inc],w,0, df['volume'][inc], fill_color="#D5E1DD", line_color="black",y_range_name="vol",fill_alpha=1)
    p.vbar(df['date'][dec],w,0, df['volume'][dec], fill_color="#F2583E", line_color="black",y_range_name="vol",fill_alpha=1)
    p.legend.location="bottom_right"
    p.legend.level='underlay'
    
    k=0   
    df1=df     
    while len(df1)>0:       
        while k<len(df1) and df1['senkou_span_a'].iloc[k]>=df1['senkou_span_b'].iloc[k] and df1['senkou_span_a'].iloc[k+1]>=df1['senkou_span_b'].iloc[k+1]:
            k+=1
        source_inc_ichi=ColumnDataSource(data=df1[:k+2]) 
        band_inc_ichi=Band(base="date", lower='senkou_span_b', upper='senkou_span_a',fill_alpha=0.15, line_width=1,
                       line_color='black',level='underlay',fill_color='green',source=source_inc_ichi)
        p.add_layout(band_inc_ichi)
        df1=df1[k+1:]
        k=0
        while k<len(df1) and df1['senkou_span_a'].iloc[k]<=df1['senkou_span_b'].iloc[k] and df1['senkou_span_a'].iloc[k+1]<=df1['senkou_span_b'].iloc[k+1]:
            k+=1
        source_dec_ichi=ColumnDataSource(data=df1[:k+2]) 
        band_dec_ichi=Band(base="date", lower='senkou_span_b', upper='senkou_span_a',fill_alpha=0.15, line_width=1,
                       line_color='black',level='underlay',fill_color='red',source=source_dec_ichi)
        p.add_layout(band_dec_ichi)
        df1=df1[k+1:]
        
        

    l0=p.line("date","SMA20",legend="SMA_20",source=source,line_dash='dashed',line_color='green',line_width=1.5)
    band = Band(base="date", lower="LB", upper="UB",fill_alpha=0.3, line_width=1, line_color='black',level='underlay',source=source)
    p.add_layout(band)
    
    l6=p.line("date","EMA50",legend="EMA_50",source=source,line_color=Category20b_20[13],line_width=1.5)
    l7=p.line("date","EMA20",legend="EMA_20",source=source,line_color=Category20b_20[1],line_width=1.5)
    
    l1=p.line("date","TEMA50",legend="TEMA_50",source=source,line_color=Category20b_20[15],line_width=1.5)
    l2=p.line("date","TEMA20",legend="TEMA_20",source=source,line_color=Category20b_20[0],line_width=1.5)
    
    l20=p.line("date","senkou_span_a",legend="Senkou_Span_a",source=source,line_color='green',line_width=0.7,line_alpha=0.5)
    l21=p.line("date","senkou_span_b",legend="Senkou_Span_b",source=source,line_color='red',line_width=0.7,line_alpha=0.8)
    l3=p.line("date",'chikou_span',legend='Chikou_span',source=source,line_color=Category20b_20[17],line_width=1.5)
    l4=p.line("date",'tenkan_sen',legend='Tenkan_sen',source=source,line_width=1.5)
    l5=p.line("date",'kijun_sen',legend='Kijun_sen',source=source,line_width=1.5,line_color="red")
    
    l8=p.circle(df["date"],df['PSAR_bear'],legend='PSAR_bear',color=Category20b_20[14],size=2.5)
    l9=p.circle(df["date"],df['PSAR_bull'],legend='PSAR_bull',color=Category20b_20[2],size=2.5)
    
    code0 = """
        if (0 in checkbox0.active) {
                l0.visible = true
        } else {
                l0.visible = false
                }
        """
        
    callback0 = CustomJS(code=code0, args={})
    checkbox0 = CheckboxGroup(labels=["SMA_20"], active=[0], callback=callback0, width=90)
    callback0.args = dict(l0=l0, checkbox0=checkbox0)
    
    code1 = """
        if (0 in checkbox1.active) {
                l1.visible = true
        } else {
                l1.visible = false
                }
        """
        
    callback1 = CustomJS(code=code1, args={})
    checkbox1 = CheckboxGroup(labels=["TEMA_50"], active=[0], callback=callback1, width=90)
    callback1.args = dict(l1=l1, checkbox1=checkbox1)
    
    code2 = """
        if (0 in checkbox2.active) {
                l2.visible = true
        } else {
                l2.visible = false
                }
        """
        
    callback2 = CustomJS(code=code2, args={})
    checkbox2 = CheckboxGroup(labels=["TEMA_20"], active=[0], callback=callback2, width=90)
    callback2.args = dict(l2=l2, checkbox2=checkbox2)
    
    code3 = """
        if (0 in checkbox3.active) {
                l3.visible = true
        } else {
                l3.visible = false
                }
        """
        
    callback3 = CustomJS(code=code3, args={})
    checkbox3 = CheckboxGroup(labels=['Chikou_span'], active=[0], callback=callback3, width=90)
    callback3.args = dict(l3=l3, checkbox3=checkbox3)
    
    code4 = """
        if (0 in checkbox4.active) {
                l4.visible = true
        } else {
                l4.visible = false
                }
        """
        
    callback4 = CustomJS(code=code4, args={})
    checkbox4 = CheckboxGroup(labels=['Tenkan_sen'], active=[0], callback=callback4, width=90)
    callback4.args = dict(l4=l4, checkbox4=checkbox4)
    
    code5 = """
        if (0 in checkbox5.active) {
                l5.visible = true
        } else {
                l5.visible = false
                }
        """
        
    callback5 = CustomJS(code=code5, args={})
    checkbox5 = CheckboxGroup(labels=['Kijun_sen'], active=[0], callback=callback5, width=90)
    callback5.args = dict(l5=l5, checkbox5=checkbox5)
    
    code6 = """
        if (0 in checkbox6.active) {
                l6.visible = true
        } else {
                l6.visible = false
                }
        """
        
    callback6 = CustomJS(code=code6, args={})
    checkbox6 = CheckboxGroup(labels=["EMA_50"], active=[0], callback=callback6, width=90)
    callback6.args = dict(l6=l6, checkbox6=checkbox6)
    
    code7 = """
        if (0 in checkbox7.active) {
                l7.visible = true
        } else {
                l7.visible = false
                }
        """
        
    callback7 = CustomJS(code=code7, args={})
    checkbox7 = CheckboxGroup(labels=["EMA_20"], active=[0], callback=callback7, width=90)
    callback7.args = dict(l7=l7, checkbox7=checkbox7)
    
    code8 = """
        if (0 in checkbox8.active) {
                l8.visible = true
        } else {
                l8.visible = false
                }
        """
        
    callback8 = CustomJS(code=code8, args={})
    checkbox8 = CheckboxGroup(labels=['PSAR_bear'], active=[0], callback=callback8, width=90)
    callback8.args = dict(l8=l8, checkbox8=checkbox8)
    
    code9 = """
        if (0 in checkbox9.active) {
                l9.visible = true
        } else {
                l9.visible = false
                }
        """
        
    callback9 = CustomJS(code=code9, args={})
    checkbox9 = CheckboxGroup(labels=['PSAR_bull'], active=[0], callback=callback9, width=90)
    callback9.args = dict(l9=l9, checkbox9=checkbox9)
        
        
    hoverp1=HoverTool(tooltips=''' 

     <div>
     <div>
        <span style="font-size:12px;font-weight:bold;color:#56">@date</span>
      </div>
     <div>
          <span style="font-size:12px;color:#765;">Volume : @volume
              </span><br>
          <span style="font-size:12px;color:#765;">Value : @value</span>
              </span><br>
          <span style="font-size:12px;color:#765;">Open : @open</span>
              </span><br>
          <span style="font-size:12px;color:#765;">Close : @close</span>
              </span><br>
          <span style="font-size:12px;color:#765;">Low : @low</span>
              </span><br>
          <span style="font-size:12px;color:#765;">High : @high</span>
              </span><br>
          <span style="font-size:12px;color:#765;">Final : @final</span>
              </span><br>
          <span style="font-size:12px;color:#765;">SAM20 : @SMA20</span>
              </span><br>
          <span style="font-size:12px;color:#765;">UB: @UB</span>
              </span><br>
          <span style="font-size:12px;color:#765;">LB: @LB</span>
      </div>
       
      </div>'''
      ,renderers=[l0]) 
    
    hoverp2=HoverTool(tooltips=''' 

     <div>
     <div>
        <span style="font-size:12px;font-weight:bold;color:#56">@date</span>
      </div>
     <div>
          <span style="font-size:12px;color:#765;">Volume : @volume
              </span><br>
          <span style="font-size:12px;color:#765;">Value : @value</span>
              </span><br>
          <span style="font-size:12px;color:#765;">Open : @open</span>
              </span><br>
          <span style="font-size:12px;color:#765;">Close : @close</span>
              </span><br>
          <span style="font-size:12px;color:#765;">Low : @low</span>
              </span><br>
          <span style="font-size:12px;color:#765;">High : @high</span>
              </span><br>
          <span style="font-size:12px;color:#765;">Final : @final</span>
              </span><br>
          <span style="font-size:12px;color:#765;">EAM20 : @EMA20</span>
              </span><br>
          <span style="font-size:12px;color:#765;">EAM50 : @EMA50</span>
              </span><br>    
          <span style="font-size:12px;color:#765;">TEMA20 : @TEMA20</span>
               </span><br>
          <span style="font-size:12px;color:#765;">TEMA50 : @TEMA50</span>
            
      </div>
       
      </div>'''
      ,renderers=[l1,l2,l6,l7]) 
    
    hoverp3=HoverTool(tooltips=''' 

     <div>
     <div>
        <span style="font-size:12px;font-weight:bold;color:#56">@date</span>
      </div>
     <div>
          <span style="font-size:12px;color:#765;">Volume : @volume
              </span><br>
          <span style="font-size:12px;color:#765;">Value : @value</span>
              </span><br>
          <span style="font-size:12px;color:#765;">Open : @open</span>
              </span><br>
          <span style="font-size:12px;color:#765;">Close : @close</span>
              </span><br>
          <span style="font-size:12px;color:#765;">Low : @low</span>
              </span><br>
          <span style="font-size:12px;color:#765;">High : @high</span>
              </span><br>
          <span style="font-size:12px;color:#765;">Final : @final</span>
              </span><br>
          <span style="font-size:12px;color:#765;">Tenkan_Sen : @tenkan_sen</span>
              </span><br>
          <span style="font-size:12px;color:#765;">Kijun_Sen: @kijun_sen</span>
              </span><br>    
          <span style="font-size:12px;color:#765;">Chikou_Span : @chikou_span</span>
               </span><br>
          <span style="font-size:12px;color:#765;">Senkou_Span_a : @senkou_span_a</span>
                </span><br>
          <span style="font-size:12px;color:#765;">Senkou_Span_b : @senkou_span_b</span>
            
      </div>    
      </div>'''
      ,renderers=[l3,l4,l5,l20,l21]) 
        
    p.add_tools(hoverp1)
    p.add_tools(hoverp2)
    p.add_tools(hoverp3)
    
    q = figure(x_range=list(df['date']), tools=TOOLS2, plot_width=1337,plot_height=180,y_range=(0,c))
    q.xaxis.major_label_orientation = np.pi/4 
    q.xaxis.major_label_text_font_size="0px"
    q.vbar(df['date'][inc],w,0, df['power'][inc], fill_color="#D5E1DD", line_color="black")
    q.vbar(df['date'][dec],w,0, df['power'][dec], fill_color="#F2583E", line_color="black")
    q.yaxis.axis_label="Power"
    q.axis.axis_label_text_font_size="15px"  
    q.toolbar.logo=None
    spanq=Span(location=1,dimension="width",line_width=0.5,line_color="black") 
    q.add_layout(spanq)

    l05 = q.line("date", "power5", color=Viridis3[0], legend="Power5",line_width=1.5,source=source)
    l10 = q.line("date","power10", color=Viridis3[1], legend="Power10",line_width=1.5,source=source)
    l15 = q.line("date", "power15", color=Viridis3[2], legend="Power15",line_width=1.5,source=source)

    hoverq=HoverTool(tooltips=''' 
            
     <div>
     <div>
        <span style="font-size:12px;font-weight:bold;color:#56">@date</span>
      </div>
     <div>
          <span style="font-size:12px;color:#765;">Power : @power
              </span><br>
          <span style="font-size:12px;color:#765;">Power5 : @power5</span>
              </span><br>
          <span style="font-size:12px;color:#765;">Power10 : @power10</span>
              </span><br>
          <span style="font-size:12px;color:#765;">Power15 : @power15</span>
              </span><br>

       </div>
       
       </div>''',renderers=[l05,l10,l15]) 

    
    q.legend.location="top_right" 
    q.add_tools(hoverq) 
    
    
    po=df['Hist']>=0
    ne=df['Hist']<0

    m=figure(x_range=list(df['date']), plot_width=1337,plot_height=120) 
    m.xaxis.major_label_orientation = np.pi/4 
    m.vbar(df['date'][po],w, 0, df['Hist'][po], fill_color="#D5E1DD", line_color="black",legend="positive")
    m.vbar(df['date'][ne],w, df['Hist'][ne], 0, fill_color="#F2583E", line_color="black",legend="negative")
    lm=m.line("date","MACD",legend="MacD",source=source)
    ls=m.line("date","Signal",legend="signal",line_color='red',source=source)  
    m.legend.location="top_right"
    m.xaxis.major_label_text_font_size="0px"
    m.yaxis.axis_label="MACD"
    m.yaxis.axis_label_text_font_size="15px"
    m.toolbar.logo=None 
    hoverm=HoverTool(tooltips=''' 
                
     <div>
     <div>
        <span style="font-size:12px;font-weight:bold;color:#56">@date</span>
      </div>
     <div>
          <span style="font-size:12px;color:#765;">MACD : @MACD
              </span><br>
          <span style="font-size:12px;color:#765;">Signal : @Signal</span>
                </span><br>
          <span style="font-size:12px;color:#765;">Difference : @Hist</span>

       </div>
       
       </div>''',renderers=[lm,ls]) 
    m.add_tools(hoverm) 
    
    a=figure(x_range=list(df['date']), plot_width=1337,plot_height=110) 
    lp=a.line("date","DIp",legend="DI+",line_width=1,source=source)
    ln=a.line("date","DIn",legend="DI-",line_color='red',line_width=1,source=source) 
    la=a.line("date","ADX",legend="Adx",line_color='black',line_width=1.3,source=source)  
    a.legend.location="top_right"
    a.yaxis.axis_label="ADX"
    a.yaxis.axis_label_text_font_size="15px" 
    a.toolbar.logo=None
    a.xaxis.major_label_text_font_size="0px"
    hovera=HoverTool(tooltips='''
                <div dir="rtl">
     <div>
     <div>
        <span style="font-size:12px;font-weight:bold;color:#56">@date</span>
      </div>
     <div>
          <span style="font-size:12px;color:#765;">DI+ : @DIp</span>
              </span><br>
          <span style="font-size:12px;color:#765;">DI- : @DIn</span>
              </span><br>
          <span style="font-size:12px;color:#765;">Adx : @ADX</span>
      </div>
       
      </div>'''
      ,renderers=[lp,ln,la]) 
         
    a.add_tools(hovera)
    
    r = figure(x_range=list(df['date']), plot_width=1337,plot_height=130)
    r.xaxis.major_label_orientation = np.pi/4 
    lr=r.line("date","RSI14",legend="RSI_14",source=source)
    span1=Span(location=30,dimension="width",line_width=1,line_color="grey") 
    span2=Span(location=70,dimension="width",line_width=1,line_color="gray") 
    span3=Span(location=50,dimension="width",line_width=0.5,line_color="grey",line_dash='dashed')
    r.add_layout(span1)
    r.add_layout(span2)
    r.add_layout(span3)
    box=BoxAnnotation(bottom=30,top=70,fill_color='red',fill_alpha=0.2)
    r.add_layout(box)
    r.legend.location="top_right"    
    r.xaxis.major_label_text_font_size="7px"
    r.yaxis.axis_label="RSI"
    r.yaxis.axis_label_text_font_size="15px" 
    r.toolbar.logo=None
    hoverr=HoverTool(tooltips=''' 
               
     <div>
     <div>
        <span style="font-size:12px;font-weight:bold;color:#56">@date</span>
      </div>
     <div>
          <span style="font-size:12px;color:#765;">RSI_14 : @RSI14

       </div>
       
       </div>''',renderers=[lr]) 
        
    r.add_tools(hoverr)
    
    lay_out=pq=row(column(checkbox0,checkbox6,checkbox7,checkbox1,checkbox2,checkbox3,checkbox4,checkbox5,checkbox8,checkbox9)
           ,column(p,q,m,a,r))
    
    output_file("D:/mohandes/our_indi/taba.html",title='taba')
    show(lay_out)

    
   

