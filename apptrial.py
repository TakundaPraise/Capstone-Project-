week_labels = [f'Week {i+1}' for i in range(12)]

    # Display the forecasts
    st.subheader("NEW PRODUCTS SALES PREDICTIONS LINE CHART ")
    #st.line_chart(rescaled_forecasts)
    chart_expander = st.expander("Click to expand", expanded=False)
    with chart_expander:
        st.write('The line chart represents the sales predictions for new fashion products. he y-axis represents the sales predictions.The chart displays a line that indicates the predicted sales over time. The values on the y-axis represent the estimated number of sales for each corresponding time point on the x-axis. The line connects these data points, showing the trend or pattern in the sales predictions. The predictions are then rescaled using normalization values and presented in both the line chart and the table.')
        week_labels = [f'Week {i+1}' for i in range(12)]

        # Create the line chart with updated labels
        fig, ax = plt.subplots(figsize=(12, 6))
        #ax.plot(rescaled_forecasts)
        ax.plot(rounded_forecasts)
        ax.set_xticks(range(len(week_labels)))
        ax.set_xticklabels(week_labels, rotation=90)
        ax.set_xlabel('Week')
        #ax.set_xticks(range(len(month_labels)))
        #ax.set_xticklabels(month_labels)
        #ax.set_xlabel('Week')
        ax.set_ylabel('Sales Predictions')
        st.pyplot(fig)
        #st.line_chart(rescaled_forecasts)
        

    st.subheader("NEW PRODUCTS SALES PREDICTIONS TABLE")
    #forecast_df = pd.DataFrame(rescaled_forecasts, columns=['Sales'])
    forecast_df = pd.DataFrame(rounded_forecasts, columns=['Sales'])
    forecast_df.index = week_labels

    #forecast_df.index = ['Jul', 'Aug','Sep', 'Oct', 'Nov','Dec','Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'  ]

# Display the forecasts
    st.subheader("NEW PRODUCTS SALES PREDICTIONS LINE CHART ")
    #st.line_chart(rescaled_forecasts)
    chart_expander = st.expander("Click to expand", expanded=False)
    with chart_expander:
        st.write('The line chart represents the sales predictions for new fashion products. he y-axis represents the sales predictions.The chart displays a line that indicates the predicted sales over time. The values on the y-axis represent the estimated number of sales for each corresponding time point on the x-axis. The line connects these data points, showing the trend or pattern in the sales predictions. The predictions are then rescaled using normalization values and presented in both the line chart and the table.')
        # Generate the week labels
        month_labels = ['January', 'February', 'March', 'April', 'May', 'June', 
                    'July', 'August', 'September', 'October', 'November', 'December']
        week_labels = []
        for i in range(12):
            month_index = i // 4
            week_index = i % 4
            week_labels.append(f"{month_labels[month_index]} Week {week_index + 1}")

        # Create the line chart with updated labels
        fig, ax = plt.subplots(figsize=(12, 6))
        #ax.plot(rescaled_forecasts)
        ax.plot(rounded_forecasts)
        ax.set_xticks(range(len(week_labels)))
        ax.set_xticklabels(week_labels, rotation=90)
        ax.set_xlabel('Week')
        #ax.set_xticks(range(len(month_labels)))
        #ax.set_xticklabels(month_labels)
        #ax.set_xlabel('Week')
        ax.set_ylabel('Sales Predictions')
        st.pyplot(fig)
        #st.line_chart(rescaled_forecasts)
        

    st.subheader("NEW PRODUCTS SALES PREDICTIONS TABLE")
    #forecast_df = pd.DataFrame(rescaled_forecasts, columns=['Sales'])
    forecast_df = pd.DataFrame(rounded_forecasts, columns=['SalesPredictions '])
    forecast_df.index = week_labels

    #forecast_df.index = ['Jul', 'Aug','Sep', 'Oct', 'Nov','Dec','Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'  ]
    #st.table(forecast_df)
    chart_expander = st.expander("Click to expand", expanded=False)
    with chart_expander:
        st.write('The table represents the sales predictions for the new fashion products. The values in the column 0 represents the months and the values in the other column represents the sales predictions of the products . The table dislpays the values of the sales corresponding to the months.')
        st.table(forecast_df)

    

