from audioop import bias
from cmath import log
import scipy
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
import matplotlib.text as mtext
from matplotlib.legend_handler import HandlerTuple
import seaborn as sns
import numpy as np
from math import sqrt, ceil
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


class NoPaddingHandler(HandlerBase):
    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        # Create an invisible Text object just to avoid IndexError
        txt = mtext.Text(0, 0, '')
        return [txt]

################################################# FOR VALIDATION AND RESIDUAL PLOTS ###############################################
###################################################################################################################################
def get_validation_plot(x, y, kde=True, suptitle=None, validation=True, joint_plot=False, theil_slope=False, log_norm=False, ax_val=None, title_val="", xlabel_val="", ylabel_val="", color_val="blue", metrics=['r2','mae','nrmse','mape','mdsa','slope', 'intercept','bias'], trendline=True, reference_line=True, residuals=False, ax_res=None, title_res="", xlabel_res="", color_res="black", res_log=False, legend_loc='upper left',reference_line_legend=True, marker_size=25, marker_border='none', category=None, separator=None, separator_legend=False, min_threshold=-100, grid=0.5, reference_line_weight=1.0, trendline_color='blue', trendline_weight=1.0):
    """
    x: true/observed value, y: predicted value
    x and y are array like, in the shape (n); where n=no of samples
    This function returns a plot with the validation fit, regression line, reference line and title for the given data
    """
    # log_norm = False
    # verify args
    if ax_val is not None and not validation:
        return "Invalid arguments - validation flag False but validation axis specified"
    elif ax_res is not None and not residuals:
        return "Invalid arguments - residuals flag False but residuals axis specified"
    elif validation and ax_val is not None and residuals and ax_res is None:
        return "Invalid arguments - residuals axis not specified but validation axis specified"
    elif residuals and ax_res is not None and validation and ax_val is None:
        return "Invalid arguments - validation axis not specified but residuals axis specified"
    
    # restrict values smaller than min threshold
    if min_threshold is not None:
        x[x<min_threshold] = min_threshold
        y[y<min_threshold] = min_threshold

    x_linear = x
    y_linear = y
    if log_norm: #CHECK 
        x, y = np.log10(x), np.log10(y)
        x[x<0], y[y<0] = 0.00001, 0.00001

    
    # fit linear model to get R2 value    
    lm = scipy.stats.linregress(x=x, y=y)
    slope, intercept = lm.slope, lm.intercept

    # # check for thiel slope flag # Thiel currently is only defined for univariate fitting. Researching ways to derive thiel slopes for multi-variate fitting
    # if theil_slope:
    #     slope, intercept, low_slope, high_slope = stats.theilslopes(y, x, 0.95, method='separate')
    #     # calculate thiel R2
    #     theil_y_pred = slope*y + intercept
    #     lm_theil = scipy.stats.linregress(x=x, y=theil_y_pred)

    # calculate residuals
    residual_values = x - y

    # evaluation stats
    r2_value = round(lm.rvalue,2)
    rmse_value = round(root_mean_squared_error(x,y),2)
    nrmse_value = round(root_mean_squared_error(x,y)/(np.max(x)-np.min(x))*100,2)
    mape_value = round(mean_absolute_percentage_error(x,y)*100,2)
    # use mdsa is log-based so works only with positive values
    mdsa_value = round(100*(np.exp(np.median(np.abs(np.log(y/x))))-1),2) # from Morely, 2018 (https://doi.org/10.1002/2017SW001669) 
    signed_bias = round(100*(np.sign(np.median(np.log(y/x))))*(np.exp(np.median(np.abs(np.log(y/x))))-1) ,2) # signed systematic bias
    bias_linear =round((np.sum(y - x)/np.sum(x))*100,2)
    mae_value = round(mean_absolute_error(x,y),2)
    # mdsa_value = round((1-np.median(2*abs(x-y)/(abs(x)+abs(y))))*100,2) # this is a different error, need to find reference
    
    # load evaluation stats in a label hash
    label_hash = {'r2':rf"$R^2={r2_value}$", 'mae':rf"$MAE={mae_value}$", 'rmse':rf"$RMSE={rmse_value}$", 'nrmse':rf"$NRMSE={nrmse_value}$%", 'mape':rf"$MAPE={mape_value}$%", 'mdsa':rf"$MdSA={mdsa_value}$%", 'bias':rf"$\beta={signed_bias}$%", 'bias_linear': rf"$\beta={bias_linear}$%", 'slope':rf"$S={round(slope,2)}$", 'intercept':rf"$c={round(intercept,2)}$"}

    # create empty container plot, set fig and axes
    fig = None
    if (ax_val is None) and (ax_res is None):
        if validation and residuals:
            # 2 plots
            fig, ax = plt.subplots(1,2, figsize=(14,7))
            ax_val = ax[0]
            ax_res = ax[1]        
        elif validation and not residuals:
            fig, ax_val = plt.subplots()
        elif not validation and residuals:
            fig, ax_res = plt.subplots()
    if log_norm:
        # set log axis
        if ax_val: 
            ax_val.set_xscale('log')
            ax_val.set_yscale('log')
        # x = np.log(x)
        # y = np.log(y)
        # x[np.isnan(x)] = 0
        # x[np.isinf(x)] = 0
        # y[np.isnan(y)] = 0
        # y[np.isinf(y)] = 0

    # set super title 
    if suptitle is not None:
        fig.suptitle(suptitle)

    # validation scatter plot
    if validation:
        # prepare label

        label = '\n'.join((label_hash[i] for i in metrics))

        # add scatter plot and R2
        # scatterplot used for base scatterplot
         
        if separator == 'hue':
            if separator_legend:
                sns.scatterplot(x=x_linear, y=y_linear, edgecolor=marker_border, hue=category, ax=ax_val, color=color_val, s=marker_size, legend=separator_legend).set(title=f"{title_val}\nN = {len(x)}", 
                                                                                 xlabel=xlabel_val, 
                                                                                 ylabel=ylabel_val)
            else:
                sns.scatterplot(x=x_linear, y=y_linear, edgecolor=marker_border, hue=category, ax=ax_val, color=color_val, s=marker_size, legend=separator_legend).set(title=f"{title_val}\nN = {len(x)}", 
                                                                                 xlabel=xlabel_val, 
                                                                                 ylabel=ylabel_val)
                # legend_text = Line2D([], [], linestyle='none', label=label)
                # ax_val.legend(handles=[legend_text])

        elif separator == 'style':
            if separator_legend:
                sns.scatterplot(x=x_linear, y=y_linear, edgecolor=marker_border, style=category, ax=ax_val, color=color_val, s=marker_size, legend=separator_legend).set(title=f"{title_val}\nN = {len(x)}", 
                                                                                 xlabel=xlabel_val, 
                                                                                 ylabel=ylabel_val)
            else:
                sns.scatterplot(x=x_linear, y=y_linear, edgecolor=marker_border, hue=category, ax=ax_val, color=color_val, s=marker_size, legend=separator_legend).set(title=f"{title_val}\nN = {len(x)}", 
                                                                                 xlabel=xlabel_val, 
                                                                                 ylabel=ylabel_val)
                # legend_text = Line2D([], [], linestyle='none', label=label)
                # ax_val.legend(handles=[legend_text])

        elif separator == 'uncertainty':
            pass
            # to be developed
            ax_val.scatter(x_linear, y_linear, c=category, cmap="RdYlGn_r", edgecolor='k', alpha=0.8)
            
        elif separator == None:
            sns.scatterplot(x=x_linear, y=y_linear, edgecolor=marker_border, ax=ax_val, color=color_val, s=marker_size, legend=False).set(title=f"{title_val}\nN = {len(x)}", 
                                                                                 xlabel=xlabel_val, 
                                                                                 ylabel=ylabel_val)
        else:
            return "Invalid argument for separator - must be either \"hue\" or \"style\"!"
        # add kde contour lines
        if kde:
            sns.kdeplot(x=x_linear, y=y_linear, color='black', log_scale=log_norm, fill=False, ax=ax_val, alpha=0.2)

        # include theil slopes
        if theil_slope:
            ax_val.plot(x, intercept + slope*x, 'k-.')
            ax_val.plot(x, intercept + low_slope*x, 'r--')
            ax_val.plot(x, intercept + high_slope*x, 'r+')
        
        # add trendline
        if trendline:
            reference_line_opacity = 0.6
            # sns.regplot(x=x, y=y, ci=95, label=label, ax=ax_val, color=color_val,).set(title=f"{title_val}\nN = {len(x)}", 
            #                                                                      xlabel=xlabel_val, 
            #                                                                      ylabel=ylabel_val)
            # regplot only used to add trendline
            # sns.regplot(x=x, y=y, ci=95, scatter=False, ax=ax_val, color='black',)

            # # extend both axis equally to match min and max
            # lims = [
            # np.min([ax_val.get_xlim(), ax_val.get_ylim()]),   # min of both axes
            # np.max([ax_val.get_xlim(), ax_val.get_ylim()]),   # max of both axes
            # ]   
        
            # # equal axes length and add 1:1 line
            # ax_val.set_xlim(lims)
            # ax_val.set_ylim(lims)

            if log_norm:
                min_val, max_val = min(np.min(x), np.min(y)), max(np.max(x), np.max(y))
                x_fit = np.logspace(min_val, max_val, 100)
                y_fit = 10 ** (slope * np.log10(x_fit) + intercept)
                ax_val.plot(x_fit, y_fit, color=trendline_color, linewidth=trendline_weight)
            else:
                xmin, xmax = ax_val.get_xlim()
                xextended = np.array([xmin, xmax])
                prediction_fit = slope*xextended + intercept
                ax_val.plot(xextended, prediction_fit, color=trendline_color, linewidth=trendline_weight)
        else:
            reference_line_opacity = 1.0
        
        if reference_line:
            # for 1:1 line
            lims = [
            np.min([ax_val.get_xlim(), ax_val.get_ylim()]),   # min of both axes
            np.max([ax_val.get_xlim(), ax_val.get_ylim()]),   # max of both axes
            ]   
        
            # equal axes length and add 1:1 line
            ax_val.set_xlim(lims)
            ax_val.set_ylim(lims)
            if reference_line_legend == True:
                # ref_label = '1:1'
                ref_line_handle = Line2D([], [], alpha=reference_line_opacity, color='black', linestyle='--', label='1:1', linewidth=reference_line_weight)
            else:
                ref_line_handle = Line2D([], [], label=None)
            if log_norm:
                ax_val.axline((1, 1), (10, 10), alpha=reference_line_opacity, zorder=0, linestyle='dashed', color='black', linewidth=reference_line_weight)
            else:
                ax_val.axline((0,0), slope=1, alpha=reference_line_opacity, zorder=0, linestyle='dashed', color='black', linewidth=reference_line_weight)

        # add legend and grid
        if not separator_legend:
            if reference_line_legend == True:
                # add metrics
                legend_text_metrics = Line2D([], [], color='none', label=label)

                # ref_label = '1:1'
                ref_line_handle = Line2D([], [], alpha=reference_line_opacity, color='black', linestyle='--', label='1:1')
                ax_val.legend(handles=[legend_text_metrics, ref_line_handle], loc=legend_loc, frameon=True, handletextpad=0.2, handlelength=1)
            else:
                legend_metrics_handle = label
                ax_val.legend(
                    handles=[legend_metrics_handle],
                    labels=[label],
                    handler_map={legend_metrics_handle: NoPaddingHandler()},
                    loc=legend_loc,
                    frameon=True,
                    handletextpad=0.2, 
                    handlelength=1
                )
                # ax_val.legend(handles=[legend_metrics_handle,], loc=legend_loc, frameon=True)

            # legend_text_metrics = Line2D([], [], linestyle='none',)
            # handles, labels = ax_val.get_legend_handles_labels()
            # handles.append(legend_text_metrics)
            # labels.append(label)
            # handles = handles[1:] + [handles[0]]
            # labels = labels[1:] + [labels[0]]
            # ax_val.legend(
            #     handles=handles,
            #     labels=labels,
            #     handler_map={tuple: HandlerTuple(ndivide=None)},
            #     loc=legend_loc,
            #     frameon=True
            # )
            # ax_val.legend(handles=[legend_text], loc=legend_loc)
        else:
            ax_val.legend(loc=legend_loc, handletextpad=0.2, handlelength=1)
        ax_val.grid(alpha=grid)
        # plt.show()

    # residuals' histogram
    if residuals:
        print('set residuals')
        sns.histplot(residual_values, log_scale=log_norm, color=color_res, kde=(not log_norm), ax=ax_res).set(title=f"{title_res}\nN={len(residual_values)}", xlabel=xlabel_res)
    # print('***********plotting finished***********')

    return fig, {'r2': r2_value, 'rmse':rmse_value, 'nrmse': nrmse_value, 'mape':mape_value, 'mae':mae_value, 'mdsa':mdsa_value, 'bias':signed_bias, 'bias_linear': bias_linear, 'residuals':residual_values, 'slope': round(slope,2), 'intercept': intercept}
###################################################################################################################################


###################################################### SCALER FUNCTION ############################################################
###################################################################################################################################
def scale(x, invert=False, llim=0, ulim=1, min=0.1, max=1000):
    """
    x: np array 
    llim = lower limit of scaled vector
    ulim = upper limit of scaled vector
    This function scales the series (x) based on the upper and lower limits. 
    For instance: if llim=0 and ulim=1; it will be a min-max scaler

    Reference: https://stats.stackexchange.com/questions/178626/how-to-normalize-data-between-1-and-1
    """
    # if type == 'min-max':
    #     llim, ulim = (0,1)
    # elif type == 'pos-neg':
    #     llim, ulim = (-1,1)
        
    if invert:
        # x_tr = (x - llim)*(np.max(x) - np.min(x))/(ulim - llim) + np.min(x)
        x_tr = (x - llim)*(max - min)/(ulim - llim) + min
    else:
        # x_tr = (ulim - llim)*(x - np.min(x))/(np.max(x) - np.min(x)) + llim
        x_tr = (ulim - llim)*(x - min)/(max - min) + llim
    
    x_tr[np.isnan(x_tr)] = 0

    return x_tr
###################################################################################################################################


############################################## FEATURE IMPORTANCE PLOTS ###########################################################
###################################################################################################################################
def get_feature_importance_plot(model_obj=None, model_name='Random Forest', ax=None, title="", xlabel="", ytick_labels=None, color="black"):
    """
    model_name is preset model name, model_obj is fitted model object
    This function returns a feature importance plot based on the fitted model (and respective features)
    """
    models = ["Random Forest", "Gradient Boosting", "Extreme Gradient Boosting"]
    
    if model_name == models[0]:
        importances = model_obj.feature_importances_
        xlabel = 'Mean Decrease in Impurity'
    elif model_name == models[1]:
        importances = model_obj.feature_importances_
        xlabel = 'Mean Decrease in Impurity'
    elif model_name == models[2]:
        # note: xgb does not include features with zero importance when using the get_score() method
        importances = model_obj.feature_importances_#get_booster().get_score(importance_type='gain')
        # importances = list(importances.values())
        xlabel = 'Information Gain'
    else:
        return f"arg \"model_name\" must be from {models}"

    # container plot
    fig = None
    if ax is None:
        fig, ax = plt.subplots()

    # feature importance plot
    sns.barplot(importances, orient='h', color=color, ax=ax).set(title=title)

    # set axes ticks and titles
    ax.set_yticks(range(len(importances)))
    if ytick_labels is not None:
        ax.set_yticklabels(ytick_labels)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    return fig
###################################################################################################################################


################################################### EMPIRICAL PC MODELS ###########################################################
###################################################################################################################################
def predict_bio_optical(rrs, model='sm12', base=400, label=False):
    """
    Takes rrs (400-800nm) and computes the phycocyanin concentration (mg/m3) based on the specified bio-optical model
    """
    if label:
        pc_predictions = {
            #'de93': (rrs.iloc[600-base] + rrs.iloc[648-base]) - rrs.iloc[624-base],
            'sc00': rrs.loc[650]/rrs.loc[624],
            'si05': rrs.loc[709]/rrs.loc[620],
            'mi09': rrs.loc[700]/rrs.loc[600],
            'mm09': rrs.loc[724]/rrs.loc[600],
            'sm12': rrs.loc[709]/rrs.loc[600],
            'mm12': (rrs.loc[708]-rrs.loc[665])/(rrs.loc[708]+rrs.loc[665]),
            'hu10': ((1/rrs.loc[615]) - (1/rrs.loc[600]))*rrs.loc[724],
            'mi14': ((1/rrs.loc[629]) - (1/rrs.loc[659]))*rrs.loc[724]
        }
    else:
        pc_predictions = {
            #'de93': (rrs.iloc[600-base] + rrs.iloc[648-base]) - rrs.iloc[624-base],
            'sc00': rrs.iloc[650-base]/rrs.iloc[625-base],
            'si05': rrs.iloc[709-base]/rrs.iloc[620-base],
            'mi09': rrs.iloc[700-base]/rrs.iloc[600-base],
            'mm09': rrs.iloc[724-base]/rrs.iloc[600-base],
            'sm12': rrs.iloc[709-base]/rrs.iloc[600-base],
            'mm12': (rrs.iloc[708-base]-rrs.iloc[665-base])/(rrs.iloc[708-base]+rrs.iloc[665-base]),
            'hu10': ((1/rrs.iloc[615-base]) - (1/rrs.iloc[600-base]))*rrs.iloc[725-base],
            'mi14': ((1/rrs.iloc[629-base]) - (1/rrs.iloc[659-base]))*rrs.iloc[724-base]
        }

    return pc_predictions[model]
###################################################################################################################################


####################################### BAND RATIOS ###############################################################################
###################################################################################################################################
def get_line_height(rrs, w0, w1, w2, base=400, df_like=True, label=False):
    """
    base, w0, w1 and w2 are integer like. The args specifiy the wavelengths.
    rrs is the dataframe containing remote sensing wavelenghts 1nm apart, of dimension nxm; where n is the number of wavelenths and m is the number of samples
    This function calculates the spectral line height using the rrs at the specified wavelengths
    Returns the line height or spectral shape (float/ array-like)
    """
    if df_like:
        if label:
            # line_height = rrs.loc[w1] - (rrs.loc[w2] + (rrs.loc[w0]-rrs.loc[w2])*((w2-w1)/(w2-w0)))
            line_height = rrs.loc[w1] - rrs.loc[w0] - (rrs.loc[w2] - rrs.loc[w0])*((w1-w0)/(w2-w0))
        else:    
            # line_height = rrs.iloc[w1-base] - (rrs.iloc[w2-base] + (rrs.iloc[w0-base]-rrs.iloc[w2-base])*((w2-w1)/(w2-w0)))
            line_height = rrs.iloc[w1-base] - rrs.iloc[w0-base] - (rrs.iloc[w2-base] - rrs.iloc[w0-base])*((w1-w0)/(w2-w0))
    else:
        # line_height = rrs[:,w1-base] - (rrs[:,w2-base] + (rrs[:,w0-base]-rrs[:,w2-base])*((w2-w1)/(w2-w0)))
        line_height = rrs[:,w1-base] - rrs[:,w0-base] - (rrs[:,w2-base] - rrs[:,w0-base])*((w2-w1)/(w2-w0))
    return line_height
###################################################################################################################################


######################################### LINE HEIGHTS ############################################################################
###################################################################################################################################
def get_band_ratio(rrs, b1, b2, base=400, label=False):
    """
    base, b1 and b2 are integer like. The args specifiy the wavelengths.
    rrs is the dataframe containing remote sensing wavelenghts 1nm apart, of dimension nxm; where n is the number of wavelenths and m is the number of samples
    This function calculates the band ratio using the rrs at the specified wavelengths
    Returns the line height (float/ array-like)
    """
    if label:
        return rrs.loc[b1]/rrs.loc[b2]
    else:
        return rrs.iloc[b1-base]/rrs.iloc[b2-base]
###################################################################################################################################

######################################### SPECTRAL ALIGNMENT ######################################################################
###################################################################################################################################
def align_sensor_SRF(insitu_rrs, weights, wavelength_min, bandwidth, base=400):
    """
    Args:
    - insitu_rrs: array like; series of rrs values in the wavelength range of the insitu sensor
    - weights: array like; spectral response function weights of the current band of the target sensor 
    - wavelength_min: int like; starting wavelength of the current band of the target sensor
    - bandwidth: int like; bandwidth of the current band of the target sensor
    - base: the base index for insitu wavelengths

    Description:
    This function takes in the insitu rrs and the spectral response function weights of the target 
    remote sensor, and returns a single value for the current band for the target sensor
    """
    slice_start = wavelength_min-bandwidth//2-base
    slice_end = wavelength_min+ceil(bandwidth/2)-base
    rrs = insitu_rrs[slice_start:slice_end]

    # here take weighted average of rrs with the weights. 
    # by definition, weighted average is the sum of weights times rrs over sum of all weights
    # return np.mean(rrs*weights)
    return np.mean(rrs*weights)
###################################################################################################################################

################################################## FEATURE SELECTION ##############################################################
def select_features(df, method='alt', auto_corr_threshold=0.95, target_corr_threshold=0.4, target_name = 'PC'):
    """
    Args:
    - df: pd.Dataframe like, the dataframe containing features and target variable data
    - method: ['target-filtered', 'feature-filtered', 'target-cross-correlation', 'mean-cross-correlation']
    - auto_corr_threshold: 1.0 if want to skip this. Removes features (covariables) having correlation more than this value
    - target_corr_threshold: 0.0 if want to skip this. Drops features (covaribales) having correlation with target less than this value
    - target_name: column name of the target variable

    Description: 
    This function takes in a pandas dataframe having target and predictor columns, removes features as per the specified
    auto-correlation and target-correlation criteria, and returns the refiend dataframe
    """
    
    if method == 'alt':
        # Preparing data
    	features_drop_list = [] # This will contain the list of features to be dropped
    	features_index_drop_list = [] # This will contain the index of features to be dropped as per df_input
    	corr_matrix = abs(df.corr())
    	corr_target = corr_matrix[target_name]
    
    	# Selecting features to be dropped (Using two for loops that runs on one triangle of the corr_matrix to avoid checking the correlation of a variable with itself)
    	for i in range(corr_matrix.shape[0]):
    		for j in range(i+1,corr_matrix.shape[0]):
    
    			# The following if statement checks if each correlation value is higher than threshold (or equal) and also ensures the two columns have NOT been dropped already.  
    			if corr_matrix.iloc[i,j]>=auto_corr_threshold and i not in features_index_drop_list and j not in features_index_drop_list:
    			
    				# The following if statement checks which of the 2 variables with high correlation has a lower correlation with target and then drops it. If equal we can drop any and it drops the first one (This is arbitrary)
    				if corr_target[corr_matrix.columns[i]] >= corr_target[corr_matrix.columns[j]]:
    					features_drop_list.append(corr_matrix.columns[j])	# Name of variable that needs to be dropped appended to list
    					features_index_drop_list.append(j)	# Index of variable that needs to be dropped appended to list. This is used to not check for the same variables repeatedly
    				else:
    					features_drop_list.append(corr_matrix.columns[i])
    					features_index_drop_list.append(i)
    
    	df_target_refined = df.drop(features_drop_list, axis=1)
    else:
        # dropping features based on auto-correlation
        df_features = df.copy()
        df_features = df_features.drop(columns=[target_name])
        corr_features = df_features.corr()
        
        mask = np.triu(np.ones_like(corr_features.abs(), dtype=bool))
        tri_auto_corr = corr_features.mask(mask)
        to_drop = [feature for feature in tri_auto_corr.columns if any(tri_auto_corr[feature]>auto_corr_threshold)]
        
        df_auto_refined = df.drop(to_drop, axis=1)
    
        # dropping features based on target correaltion
        
        corr = df_auto_refined.corr()
        cor_target = abs(corr[target_name])
        filtered_features = cor_target[cor_target>target_corr_threshold]
        
        df_target_refined = df_auto_refined[filtered_features.keys()]

    corr = df_target_refined.corr()
    cor_target = abs(corr[target_name])
    filtered_features = cor_target[cor_target>target_corr_threshold]
    
    df_target_refined = df_target_refined[filtered_features.keys()]    

    return df_target_refined

    
###################################################################################################################################

######################################### TRAINING PYTORCH NN #####################################################################
class linearRegression(nn.Module): # all the dependencies from torch will be given to this class [parent class] # nn.Module contains all the building block of neural networks:
    def __init__(self, input_dim):
        super(linearRegression, self).__init__()  # building connection with parent and child classes
        self.fc1=nn.Linear(input_dim,10)          # hidden layer 1
        self.fc2=nn.Linear(10,5)                  # hidden layer 2
        # self.fc3=nn.Linear(5,3)                 # hidden layer 3
        self.fc4=nn.Linear(5,1)                   # last layer
        self.double()                             # for matching torch's float32

    def forward(self,d):
        out=torch.relu(self.fc1(d))              # input * weights + bias for layer 1
        out=torch.relu(self.fc2(out))            # input * weights + bias for layer 2
        # out=torch.relu(self.fc3(out))          # input * weights + bias for layer 3
        out=self.fc4(out)                        # input * weights + bias for last layer
        return out                               # final outcome
###################################################################################################################################
def train_neural_network(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, epochs=1000, batch_size=10, lr=0.01):
    """
    Args:
    - X_train_scaled: training feature vector
    - y_train_scaled: training target vector
    - X_test_scaled: testing feature vector
    - y_test_scaled: testing target vector
    - epochs: number of epochs to train for
    - batch_size: number of samples for each iteration
    - lr: learning rate to adopt while training
    """
    # instantiate model
    input_dim = X_train_scaled.shape[1] # set input dimensions for first layer
    torch.manual_seed(42)  # to make initilized weights stable:
    model = linearRegression(input_dim)
    
    # select loss and optimizers and epochs
    batch_start = torch.arange(0, len(X_train_scaled), batch_size)
    loss_fn = nn.MSELoss() # loss function
    optimizers = optim.Adam(params=model.parameters(), lr=lr)

    # training the network
    for i in range(epochs):
        # give the input data to the architecure
        for start in batch_start:
            X_batch = X_train_scaled[start:start+batch_size]
            y_batch = y_train_scaled[start:start+batch_size]
            y_train_prediction = model(torch.from_numpy(X_batch))  # model initilizing
            loss_value = loss_fn(y_train_prediction.squeeze(), torch.from_numpy(y_batch).squeeze())   # find the loss function:
            optimizers.zero_grad()  # make gradients zero for every iteration so next iteration it will be clear
            loss_value.backward()  # back propagation
            optimizers.step()  # update weights in NN
            
    # if (i+1)%100 == 0:
    # # print training loss for every 10 epochs
    #     print(f"[epoch:{i+1}]: Training loss = {loss_value};") # LR: {before_lr} --> {after_lr}")

    return model # return trained model
###################################################################################################################################

######################################### TRAINING ML MODELS ######################################################################
###################################################################################################################################
def train_ml_models(X, y, feature_names, scaler=None, log_transform=False, log_validate=False, change_X=False, invert_predictions=False, split_shuffle=True, test_size=0.3, reference_line=False, ensemble=None, top_n=6, metrics=[], font_size=24):
    """
    Args:
    - X: numpy.ndarray like; feature vector
    - y: numpy.ndarray like; target vector
    - standardize: bool like; flag to standardize the data
    - test_size: [0,1], float like; the proportion of test data from entire data
    - ensemble: string like; flag to perform ensembling of all models; ['mean', 'median', 'weighted', 'individual', 'bagging', None]
    - feature_names: array like; list of strings wherein each string is a feature name
    - top_n: int like; number of top-performing models to select for ensembling. Default is 6 so all models are selected by default

    Description:
    This function trains the following ML models on the given data: 
    Random Forest; Gradient Boosting; Extreme Gradient Boosting; Adaboost; Multi-Layer Perceptron; 2-layer Neural Network
    And returns validation statistics and plots in a tuple.

    Return format:
    if ensemble is not None
    (models_dict, combined_plots, ensemble_stats, ensemble_plots)
    else
    (models_dict, combined_plots)
    """
    # log transform flag
    if log_transform:
        y = np.log(y+1)
        y[np.isnan(y)] = 0
        y[np.isinf(y)] = 0
        if change_X:
            X = np.log(X+1)
            X[np.isnan(X)] = 0
            X[np.isinf(X)] = 0

    # dictionary to store all info
    # format: {model_name: [model_obj, model_predictions, validation_plots, validation_stats, color]}
    ml_preds = {
        'Random Forest': [RandomForestRegressor(max_depth=3, random_state=7), None, [], None, 'green'],
        'Gradient Boosting': [GradientBoostingRegressor(random_state=7), None, [], None, 'purple'],
        'Extreme Gradient Boosting': [XGBRegressor(random_state=7, objective='reg:squarederror',), None, [], None, 'brown'],
        # 'Adaboost Regression': [AdaBoostRegressor(random_state=7, loss='linear', learning_rate=0.1), None, [], None, 'red'], # Removing adaboost due to insensitivity
        'Multi-Layer Perceptron': [MLPRegressor(random_state=7, max_iter=1000), None, [], None, 'indigo'],
        # 'Light GBM': []
        #'Neural Regression': [None, None, [], None, 'blue']
            }

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=split_shuffle, random_state=777)

    # Note: Data is scaled separately for train and test set to avoid data leakage,
    # and to avoid influence of test set on train set leading to potential overfitting
    # Guideline for standardizing/scaling data: https://datascience.stackexchange.com/questions/63717/how-to-use-standardization-standardscaler-for-train-and-test
    if scaler:
        if change_X:
            X_train_scaled = scale(X_train, type=scaler)
            X_test_scaled = scale(X_test, type=scaler)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        if scaler == 'robust':
            robust_scaler = RobustScaler()
            y_train_scaled = robust_scaler.fit_transform(y_train)
            y_test_scaled = robust_scaler.transform(y_test)
        else:
            y_train_scaled = scale(y_train, type=scaler)
            y_test_scaled = scale(y_test, type=scaler)
    else:
        # no scaling
        X_train_scaled = X_train
        X_test_scaled = X_test
        y_train_scaled = y_train
        y_test_scaled = y_test
        # X_scaler = MinMaxScaler()
        # X_train_scaled = X_scaler.fit_transform(X_train)
        # X_test_scaled = X_scaler.transform(X_test)
        # X_train_scaled[np.isnan(X_train)] = 0
        # X_test_scaled[np.isnan(X_test)] = 0
        
        # Y_scaler = MinMaxScaler()
        # y_train_scaled = Y_scaler.fit_transform(y_train)
        # y_test_scaled = Y_scaler.transform(y_test)
        # y_train_scaled[np.isnan(y_train)] = 0
        # y_test_scaled[np.isnan(y_test)] = 0

    # training ML models
    for model in ml_preds:
        model_obj = None
        predicted = None
        # skip neural network because it uses a separate function to train
        if model == 'Neural Regression':
            model_obj = train_neural_network(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, epochs=1000, batch_size=10, lr=0.001)

            # get inferences on test data
            with torch.no_grad():
                model_obj.eval()   # make model in evaluation stage
                y_test_prediction=model_obj(torch.from_numpy(X_test_scaled))
            predicted = y_test_prediction.numpy().reshape(-1,1)
            
        else:
            # 1. fit model
            model_obj = ml_preds[model][0]
            model_obj = model_obj.fit(X_train_scaled, y_train_scaled.ravel())

            # 2. get inferences on test data
            predicted = model_obj.predict(X_test_scaled).reshape(-1,1)

        # for all models - save the trained model object
        ml_preds[model][0] = model_obj

        # for all models - replace all negative inferences with 0
        predicted[predicted<0]=0

        # for all models - check inverse transformation flag
        if invert_predictions:
            # first invert scaling if scaled
            if scaler:
                if scaler == 'robust':
                    predicted = robust_scaler.inverse_transform(predicted)
                else:
                    predicted = scale(predicted, type=scaler, invert=True)
                # y_test is not scaled
            # then inverse transform if transformed 
            if log_transform:
                predicted = np.exp(predicted)-1
                y_test = np.exp(y_test)-1
                y_test[np.isnan(y_test)] = 0
                y_test[np.isinf(y_test)] = 0
                predicted[np.isnan(predicted)] = 0
                predicted[np.isinf(predicted)] = 0
        else:
            y_test = y_test_scaled

        # save model predictions
        ml_preds[model][1] = predicted

        # save validation plots
        _, ml_preds[model][3] = get_validation_plot(x=y_test.ravel(), 
                                                    y=ml_preds[model][1].ravel(),
                                                    validation=True,
                                                    log_norm=log_validate, 
                                                    color_val=ml_preds[model][-1],
                                                    metrics=metrics,
                                                    reference_line=reference_line,
                                                    title_val=f"{model} Testing",
                                                    xlabel_val=r"insitu PC $(mg/m^3)$",
                                                    ylabel_val=r"modelled PC $(mg/m^3)$",
                                                    residuals=True,
                                                    color_res='black',
                                                    xlabel_res="Residual Value",)
        ml_preds[model][2].append(_)

        # 3. feature importances
        fi = None
        if model in ['Random Forest', 'Gradient Boosting', 'Extreme Gradient Boosting', 'Adaboost Regression']:
            fi = get_feature_importance_plot(model_obj, model, title=f"{model} Feature Importance", color=ml_preds[model][-1], ytick_labels=feature_names)
            ml_preds[model][2].append(fi)
    ###### end training ####################################
    
    # combined validation and residual plots
    ml_models = list(ml_preds.keys())
    BASE_SIZE = 7.5
    NO_OF_MODELS = len(ml_models)
    fig, ax = plt.subplots(2,NO_OF_MODELS, figsize=(NO_OF_MODELS*BASE_SIZE,2*BASE_SIZE))
    
    for col in range(NO_OF_MODELS):
        model = ml_models[col]
        get_validation_plot(x=y_test.ravel(), 
                            y=ml_preds[model][1].ravel(), 
                            validation=True,
                            log_norm=log_validate, 
                            color_val=ml_preds[model][-1],#"black", # ml_preds[model][-1],
                            metrics=metrics,
                            reference_line=reference_line,
                            title_val=f"{model} Testing",
                            xlabel_val=r"insitu PC $(mg/m^3)$",
                            ylabel_val=r"modelled PC $(mg/m^3)$",
                            ax_val=ax[0][col],
                            residuals=True,
                            ax_res=ax[1][col],
                            color_res=ml_preds[model][-1],#"black", #ml_preds[model][-1],
                            title_res=f"Histogram of Residuals",
                            xlabel_res="Residual Value")
    
    plt.rc('font', size=font_size)
    plt.rc('axes', titlesize=font_size)
    plt.rc('axes', labelsize=font_size)
    plt.rc('xtick', labelsize=font_size)
    plt.rc('ytick', labelsize=font_size)
    fig.tight_layout()

    # check for ensembling flag and criterion
    if top_n>0 or ensemble == 'weighted':
        # calculate weights 
        raw_weights = []
        for model in ml_models:
            raw_weights.append(ml_preds[model][-2]['r2']/ml_preds[model][-2]['rmse'])
        # normalize weights
        weights = []
        for weight in raw_weights:
            weights.append((weight-min(raw_weights))/(max(raw_weights)-min(raw_weights)))
        del raw_weights

    ensemble_predicted = None
    if ensemble is not None:
        # sort the top-n models as per the weights; if top-n=6; all models will be included
        ensemble_models = [list(ml_preds.keys())[i] for i in np.argsort(weights)[::-1][:top_n]]
        if ensemble == 'mean':
            ensemble_predicted = np.mean([ml_preds[model][1] for model in ensemble_models], axis=0)
        elif ensemble == 'median':
            ensemble_predicted = np.median([ml_preds[model][1] for model in ensemble_models], axis=0)
        elif ensemble == 'zhang': # based on Zhang et al, 2019; eq-1; doi: https://doi.org/10.1016/j.isprsjprs.2019.01.006
            ensemble_predicted = np.sum([ml_preds[model][1]*ml_preds[model][-2]['r2'] for model in ensemble_models], axis=0)/np.sum([ml_preds[model][-2]['r2'] for model in ensemble_models])
        elif ensemble == 'weighted':
            ensemble_predicted = np.mean([weights[i]*ml_preds[list(ml_preds.keys())[i]][1] for i in np.argsort(weights)[::-1][:top_n]], axis=0)
        elif ensemble == 'individual': # this approach is valid ONLY for selecting models when you have training data; this cannot be applied to real data
            # NOTE: evaluate every inference individually, select the best model for each inference based on (prediceted-actual).
            # NOTE: a single set of predictions can come from multiple models

            # combine all predictions - these are all model preds with shape (no_of_samples, no_of_models)
            combined_predictions = np.array([ml_preds[model][1].ravel() for model in ml_preds.keys()]).transpose()
            
            # calculate residuals
            combined_residuals = combined_predictions - y_test

            # gather indices of the best model based on residuals, for all predictions
            indices = np.argmin(abs(combined_residuals), axis=1)

            # extract the best predictions for each inference, from all predictions
            ensemble_predicted = combined_predictions[np.arange(len(combined_predictions)), indices]

            # delete local variables
            del combined_predictions, combined_residuals, indices
        elif ensemble == 'bagging':
            # combine the predictions of all (selected) models. Models will be selected based on 'top_n' argument.
            combined_predictions = np.array([ml_preds[model][1].ravel() for model in ensemble_models]).transpose()

            # use a decision tree/xgb regressor to bag all predictions
            # bagger = DecisionTreeRegressor(random_state=7, )
            bagger = XGBRegressor(random_state=7, objective='reg:squarederror',)
            
            # train data - predictions from all models, test data - in-situ values
            # bagging will be performed on the 451 points for testing
            # train on 80% (361 points), test on 20% (90 points)
            bagger.fit(combined_predictions[:361,], y_test[:361,])
            # bagger.fit(combined_predictions[:523,], y_test[:523,])

            # use the bagger to predict on the 90 test points. This shows the cumulative performance of all models
            # ensemble_predicted = bagger.predict(combined_predictions[-131:,]) # cut=90
            # y_test = y_test[-131:,] #cut=90
            ensemble_predicted = bagger.predict(combined_predictions[-90:,]) # cut=90
            y_test = y_test[-90:,] #cut=90
        else:
            print(f"Invalid args for ensemble \"{ensemble}\"")
            
        ensemble_plot, ensemble_stats = get_validation_plot(x=y_test.ravel(), 
                                                            y=ensemble_predicted.ravel(),
                                                            validation=True,
                                                            log_norm=log_validate, 
                                                            color_val='black',
                                                            metrics=metrics,
                                                            reference_line=reference_line,
                                                            title_val=f"Ensemble ({ensemble}) Testing",
                                                            xlabel_val=r"insitu PC $(mg/m^3)$",
                                                            ylabel_val=r"predicted PC $(mg/m^3)$",
                                                            residuals=True,
                                                            color_res='black',
                                                            xlabel_res="PC $(mg/m^3)$")

    if ensemble is not None:
        return (ml_preds, fig, [ensemble_models, ensemble_stats, ensemble_plot])
    else:
        return (ml_preds, fig)        
###################################################################################################################################