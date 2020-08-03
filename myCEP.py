import numpy as np
import pandas as pd
from tables import A2, A3, B3, B4, D3, D4 
import matplotlib.pyplot as plt

# qqStaQuaControl
This package can help you to build a statistical quality control chart using python lenguage

class CEP():
    def xbarChartIndividual(self,TemporalData, TotalData= None):
        '''
        :param TemporalData: Data to define the optimal condition to config the control limits
        :param TotalData: Data to define the Operational Control
        :return: LCL,center, UCL
        '''
        from tables import d2

        x = TemporalData.iloc[:,1].to_numpy()
        
        xbar = x.mean()

        MR = np.arange(0,len(TemporalData),1)
        i=0


        MR = [ ]
        while i <= len(TemporalData)-1:

            if i>=1:
                MR.append(abs(x[i] - x[i - 1]))

            i+=1
        
        n_offset = 2
        MR= np.array(MR)
        MRbar= MR.mean()

        UCL_Val = xbar + 3*(MRbar / d2[n_offset+1])
        center_Val = xbar
        LCL_Val = xbar - 3*(MRbar / d2[n_offset+1])

        LCL = []
        center = []
        UCL =[]


        if TotalData != None:
            total = TotalData.iloc[:, 1]
            for i in range(len(total)):
                UCL.append(UCL_Val)
                center.append(center_Val)
                LCL.append(LCL_Val)

        else:

            for i in range(len(x)):
                UCL.append(UCL_Val)
                center.append(center_Val)
                LCL.append(LCL_Val)

        # LCL = pd.DataFrame(data = LCL)
        # center = pd.DataFrame(data = center)
        # UCL = pd.DataFrame(data=UCL)

        return LCL,center, UCL

    def xbarChartIndividual_plot(self,TemporalData,show=False,title=None):


        lcl,center,ucl = self.xbarChartIndividual(TemporalData)


        import numpy as np
        x = TemporalData.iloc[:,0]
        x = TemporalData.iloc[:,0]
        y = TemporalData.iloc[:,1] 
        
        lcl_ = lcl[0]
        center_ = center[0]
        ucl_ = ucl[0]

    
        slower = np.ma.masked_where(y >= lcl_, y)
        smiddle = np.ma.masked_where((y <= lcl_) | (y >= ucl_), y)
        supper = np.ma.masked_where(y <= ucl_, y)

        fig = plt.figure()
        ax = fig.gca()

        ax.plot(x, y, '-',c="black")
        ax.plot(x, supper, 'sr',x, ucl,c="red")
        ax.plot(x, smiddle,'s',x, center,c="blue")
        ax.plot(x, slower, 'sr',x, lcl,c="red")

        if title==None:
            plt.title(" Gráfico - Controle Estatístico de Processos")
        else:
            plt.title(title)

        if show==True:
            plt.show()

        return plt

    def subgroup_build(self,dataX,n):
        # define o tamanho dos vetores
        Xlen = len(dataX)-1
    
        values_left = Xlen % n
        values_toProcess = Xlen - values_left
        total_subgroups = values_toProcess / n

        # define os vetores vazios
        x_samples = []
        X_bar = []
        j = 0
        while j <= values_toProcess - 1:
            i = 0
            interX = []
            while i <= n - 1:
                interX.append(dataX[j])
                i += 1
                j += 1

            X_bar.append(np.array(interX).mean())
            x_samples.append(interX)

        return x_samples#, X_bar

    def X_samples(self, subgroups):
        X_sample = []
        i = 0
        for i in subgroups:

            X_sample.append(np.mean(i))
        #return pd.DataFrame(np.arrya(np.array(r).T))
        return X_sample

    def X_barbar(self,X_bar_array):

        return np.sum(X_bar_array)/len(X_bar_array)

    def R_samples(self, X_bar_array):
        R_samples = []
        i = 0
        for i in X_bar_array:
            rMax = np.amax(i)
            rMin = np.amin(i)
            R_samples.append(rMax - rMin)
        #return pd.DataFrame(np.arrya(np.array(r).T))
        return R_samples

    def R_bar_samples(self,X_bar_array):

        return np.sum(np.array(X_bar_array))/len(X_bar_array)

    def XbarChart_XR(self,TemporalDataX, TemporaLdataY,n):

        #Construção de Subgrupos a partir de dados temporais
        #subgroupX, mediaSubgroupX = self.subgroup_build(TemporalDataX, n)
        #subgroupY, mediaSubgroupY = self.subgroup_build(TemporaLdataY, n)

        subgroupX = self.subgroup_build(TemporalDataX.to_numpy(), n)
        subgroupY = self.subgroup_build(TemporaLdataY.to_numpy(), n)

        mediaSubgroupX = self.X_samples(subgroupX)
        mediaSubgroupY = self.X_samples(subgroupY)

        r_samples= self.R_samples(subgroupY)
        #Calculos intermediários7
        x_barbar = self.X_barbar(mediaSubgroupY)
        r_bar = myCep.R_bar_samples(r_samples)

        UCL_Val = x_barbar + A2[n]*r_bar
        center_Val = x_barbar
        LCL_Val = x_barbar - A2[n]*r_bar

        LCL = []
        center = []
        UCL = []

        for i in range(len(mediaSubgroupX)):
            UCL.append(UCL_Val)
            center.append(center_Val)
            LCL.append(LCL_Val)

        return mediaSubgroupX,mediaSubgroupY, LCL,center, UCL

    def RbarChart(self,TemporalDataX, TemporaLdataY,n):

        #Construção de Subgrupos a partir de dados temporais
        subgroupX  = self.subgroup_build(TemporalDataX.to_numpy(), n)
        subgroupY  = self.subgroup_build(TemporaLdataY.to_numpy(), n)
        #subgroupX, mediaSubgroupX = myCep.subgroup_build(TemporalDataX, n)
        #subgroupY, mediaSubgroupY = myCep.subgroup_build(TemporaLdataY, n)

        mediaSubgroupX = self.X_samples(subgroupX)
        mediaSubgroupY = self.X_samples(subgroupY)

        amplitudeSubgroup= self.R_samples(subgroupY)
        #Calculos intermediários
        x_barbar = self.X_barbar(mediaSubgroupY)
        r_bar = myCep.R_bar_samples(amplitudeSubgroup)

        UCL_Val = D4[n]*r_bar
        center_Val = r_bar
        LCL_Val = D3[n]*r_bar

        LCL = []
        center = []
        UCL = []

        for i in range(len(mediaSubgroupX)):
            UCL.append(UCL_Val)
            center.append(center_Val)
            LCL.append(LCL_Val)

        return mediaSubgroupX,amplitudeSubgroup, LCL,center, UCL

    def XbarRbarChart_plot(self,TemporalData,TemporalDataLimiteControl,n,title=None):
        """
        TemporalData: Pandas DataFrame 
            coluna 1 - dados de timestamp
            coluna 2 - dados de numéricos

        TemporalDataLimiteControl:
            coluna 1 - dados de timestamp
            coluna 2 - dados de numéricos
        
        """

        # Contrução de subgrupo dos Dados de Padrão p construção de cartas 
        # e Dados para Monitoramento
        #SubgroupToMonitoring = myCep.subgroup_build(TemporalData.iloc[:,1].to_numpy(),n)

        xbarout, Xout, xbarlcl, xbarcenter, xbarucl = myCep.XbarChart_XR(TemporalData.iloc[:,0],TemporalData.iloc[:,1], n=n)

        #plt = myCep.plot(xout, Xout, lcl, center, ucl, 1, 'Xbar: ' + str(i))

        rbarxout, Rout, rbarlcl, rbarcenter, rbarucl = myCep.RbarChart(TemporalData.iloc[:,0],TemporalData.iloc[:,1], n=n)

        #plt = myCep.plot(xout, Rout, lcl, center, ucl, 1, 'Rbar: ' + str(i))

        import matplotlib.pyplot as plt
        import numpy as np

        y = Xout
        lcl_ = xbarlcl[0]
        center_ = xbarcenter[0]
        ucl_ = xbarucl[0]

        supper1 = np.ma.masked_where(y <= ucl_, y)
        slower1= np.ma.masked_where(y >= lcl_, y)
        smiddle1 = np.ma.masked_where((y <= lcl_) | (y >= ucl_), y)

        y = Rout
        lcl_ = rbarlcl[0]
        center_ = rbarcenter[0]
        ucl_ = rbarucl[0]

        supper2 = np.ma.masked_where(y <= ucl_, y)
        slower2= np.ma.masked_where(y >= lcl_, y)
        smiddle2 = np.ma.masked_where((y <= lcl_) | (y >= ucl_), y)

        plt.figure()
        plt.subplot(211)
        plt.title('Gráfico  X-barra',fontsize = 9)
        plt.plot(xbarout, Xout, '-', c="black",linewidth=0.3)
        plt.plot(xbarout, supper1, 'sr', xbarout, xbarucl, c="red",linewidth=0.5,markersize=5)
        plt.plot(xbarout, smiddle1, 's', xbarout, xbarcenter, c="blue",linewidth=0.5,markersize=5)
        plt.plot(xbarout, slower1, 'sr', xbarout, xbarlcl, c="red",linewidth=0.5,markersize=5)
        plt.subplots_adjust(hspace=.3)

        plt.subplot(212)
        plt.title('Gráfico  R-barra',fontsize = 9)
        plt.xlabel('Amostras')
        plt.plot(rbarxout, Rout, '-', c="black",linewidth=0.5,markersize=5)
        plt.plot(rbarxout, supper2, 'sr', rbarxout, rbarucl, c="red",linewidth=0.5,markersize=5)
        plt.plot(rbarxout, smiddle2, 's', rbarxout, rbarcenter, c="blue",linewidth=0.5,markersize=5)
        plt.plot(rbarxout, slower2, 'sr', rbarxout, rbarlcl, c="red",linewidth=0.5,markersize=5)

        #plt.matplotlib.pyplot.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)[source]
        plt.show()



    def S_samples(self, X_bar_array):
        s = []
        i = 0
        for i in X_bar_array:
            s.append(np.std(i))

        return s

    def S_bar_samples(self, s_samples_array):

        return np.sum(np.array(s_samples_array)) / len(s_samples_array)

    def XbarChart_XS(self, TemporalDataX, TemporaLdataY, n):


        # Construção de Subgrupos a partir de dados temporais
        # subgroupX, mediaSubgroupX = self.subgroup_build(TemporalDataX, n)
        # subgroupY, mediaSubgroupY = self.subgroup_build(TemporaLdataY, n)

        subgroupX = self.subgroup_build(TemporalDataX.to_numpy(), n)
        subgroupY = self.subgroup_build(TemporaLdataY.to_numpy(), n)

        mediaSubgroupX = self.X_samples(subgroupX)
        mediaSubgroupY = self.X_samples(subgroupY)

        x_barbar = self.X_barbar(mediaSubgroupY)
        s_samples = self.S_samples(subgroupY)
        s_bar = myCep.S_bar_samples(s_samples)

        UCL_Val = x_barbar + A3[n] * s_bar
        center_Val = x_barbar
        LCL_Val = x_barbar - A3[n] * s_bar

        LCL = []
        center = []
        UCL = []

        for i in range(len(mediaSubgroupX)):
            UCL.append(UCL_Val)
            center.append(center_Val)
            LCL.append(LCL_Val)

        return mediaSubgroupX, mediaSubgroupY, LCL, center, UCL

    def SbarChart(self, TemporalDataX, TemporaLdataY, n):

        # Construção de Subgrupos a partir de dados temporais
        subgroupX = self.subgroup_build(TemporalDataX.to_numpy(), n)
        subgroupY = self.subgroup_build(TemporaLdataY.to_numpy(), n)

        mediaSubgroupX = self.X_samples(subgroupX)
        mediaSubgroupY = self.X_samples(subgroupY)

        # média de cada subgrupo
        s_samples = self.S_samples(subgroupY)

        # média intermediário
        s_bar = myCep.S_bar_samples(s_samples)

        UCL_Val = B4[n] * s_bar
        center_Val = s_bar
        LCL_Val =B3[n] * s_bar

        LCL = []
        center = []
        UCL = []

        for i in range(len(mediaSubgroupX)):
            UCL.append(UCL_Val)
            center.append(center_Val)
            LCL.append(LCL_Val)

        return mediaSubgroupX, s_samples, LCL, center, UCL

    def XbarSbarChart_plot(self,TemporalData, TemporalDataControlLimite,n,title=None):

        xout1, yout1, lcl1, center1, ucl1 = myCep.XbarChart_XS(TemporalData.iloc[:,0],TemporalData.iloc[:,1], n=n)

        xout2, yout2, lcl2, center2, ucl2 = myCep.SbarChart(TemporalData.iloc[:,0],TemporalData.iloc[:,1], n=n)

        import matplotlib.pyplot as plt
        import numpy as np

        # Gráfico X
        y = yout1
        lcl_ = lcl1[0]
        center_ = center1[0]
        ucl_ = ucl1[0]

        supper1 = np.ma.masked_where(y <= ucl_, y)
        slower1= np.ma.masked_where(y >= lcl_, y)
        smiddle1 = np.ma.masked_where((y <= lcl_) | (y >= ucl_), y)


        # Gráfico S
        y = yout2
        lcl_ = lcl2[0]
        center_ = center2[0]
        ucl_ = ucl2[0]

        supper2 = np.ma.masked_where(y <= ucl_, y)
        slower2= np.ma.masked_where(y >= lcl_, y)
        smiddle2 = np.ma.masked_where((y <= lcl_) | (y >= ucl_), y)
        plt.figure()
        plt.subplot(211)
        plt.title('Gráfico  X-barra',fontsize = 9)
        plt.plot(xout1, yout1, '-', c="black",linewidth=0.3)
        plt.plot(xout1, supper1, 'sr', xout1, ucl1, c="red",linewidth=0.5,markersize=5)
        plt.plot(xout1, smiddle1, 's', xout1, center1, c="blue",linewidth=0.5,markersize=5)
        plt.plot(xout1, slower1, 'sr', xout1, lcl1, c="red",linewidth=0.5,markersize=5)
        plt.subplots_adjust(hspace=.3)

        plt.subplot(212)
        plt.title('Gráfico  S-barra',fontsize = 9)
        plt.xlabel('Amostras')
        plt.plot(xout2, yout2, '-', c="black",linewidth=0.5,markersize=5)
        plt.plot(xout2, supper2, 'sr', xout2, ucl2, c="red",linewidth=0.5,markersize=5)
        plt.plot(xout2, smiddle2, 's', xout2, center2, c="blue",linewidth=0.5,markersize=5)
        plt.plot(xout2, slower2, 'sr', xout2, lcl2, c="red",linewidth=0.5,markersize=5)

        #plt.matplotlib.pyplot.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)[source]
        plt.show()

    def histogramPlot(self,TemporaLdataY):
        TemporaLdataY = np.array(TemporaLdataY)
        print(type(TemporaLdataY))
        #a = np.hstack(x=TemporaLdataY, bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
        plt.title("Histograma dos dados de Processo")
        n, bins, patches=plt.hist(x=TemporaLdataY, bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)

        n, bins, patches
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Valor')
        plt.ylabel('Frequência')
        plt.title('Histogram')
        plt.text(23, 45, r'$\mu=15, b=3$')
        maxfreq = n.max()
        return


"""
    Definição de dados para análise
"""

#---------------- Leitura dos dados em DataFrame
tagName = "tag_05.csv"
tagPath = '../tags/ok/02/ZT57D078.MV/' + tagName
df = pd.read_csv(tagPath,header=None, float_precision='high', dtype='object')


# Definação dos tipos dados
df.columns = ['time', 'H']
df['H'] = pd.to_numeric(df['H'], errors='coerce')
df['time'] = pd.to_datetime(df['time'], errors='coerce')


#---------------- Filtragem de intervalos de tempos
filtroTemporal = 1
if filtroTemporal == 1:
    start = '14-Nov-19 00:00:20 '
    end = '15-Nov-19 00:00:20'

    # define a variável com o intervalo especificado
    mask  = (df.iloc[:,0]>start) & (df.iloc[:,0]<=end)
    df  = df.loc[mask]


#---------------- C
df['time'] = df['time'].values.astype('float64')
#df['time'] = pd.to_timedelta(df['time']).dt.total_seconds().astype(int)


#---------------- Dados p/ Construção dos Limites de Controle
mypercent = 0.5
perc = int(len(df)*mypercent)

#---------------- Objeto com gráficos de controle
myCep = CEP()

#---------------- Dados para definição dos limites de controle
dg = df.iloc[:perc]


"""
    Construção simplificada de análise de cartas de controle
    entrada: 
        DF Bruto
        DF para construção das cartas de controle
        N° de elementos em cada subgrupo
        Qual análise estatísticas
        
    Saída: 
        Graficos de controle

"""
# Puxa cartas disponíveis
controlchart = ['XbarRbarChart','XbarSbarChart_plot','xbarChartIndividual','histogramPlot']

# opção escolhida pelo usuário - index
mychoiceIndex = 0

# opção escolhida pelo usuário - nome 
mychoice = controlchart[mychoiceIndex]

if  mychoice == controlchart[0]:

    # Gráficos de 
    f0 = myCep.XbarRbarChart_plot(df,dg,n=3)

elif mychoice == controlchart[1]:

    # Calculos intermediários 
    f1 = myCep.XbarSbarChart_plot(df,n=3)


elif mychoice == controlchart[2]:

    # Gráfico de X bar - Medidas individuais 
    f2 = myCep.xbarChartIndividual_plot(df,show=True)

elif mychoice == controlchart[3]:

    # Gráfico Histograma 
    f3 = myCep.histogramPlot(y)


'''
    Estratégia 02
    Construção de Subgrupos a partir de dados temporais
'''
# subgroup = myCep.subgroup_build(y,3)
# xx = myCep.X_samples(subgroup)


# i = 3
# nn = 3
# while i <= nn:
#     xout,Xout,lcl,center,ucl = myCep.XbarChart_XR(x,y,n=i)
#     plt = myCep.plot(xout,Xout,lcl,center,ucl,1,'Xbar: ' + str(i))

#     xout,Rout,lcl,center,ucl = myCep.RbarChart(x,y,n=i)
#     plt = myCep.plot(xout,Rout,lcl,center,ucl,1,'Rbar: ' + str(i))

#     i += 1



