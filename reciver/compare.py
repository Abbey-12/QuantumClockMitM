import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson

# # File path
# file_path1 = "/home/abebu/SimQN/security/QuantumClockMitM/reciver/data/received_data_20240715_102514_127.0.0.1.csv"
# file_path2 = "/home/abebu/SimQN/security/QuantumClockMitM/sender/data/arrivals_A.csv"

# # Read the CSV file
# df = pd.read_csv(file_path)

# # Assuming the first column contains the time data
# column_to_plot = df.columns[0]

# # Total time and bin size
# total_time = 0.14  # 0.14 seconds
# bin_size = 1e-3   # 1 microsecond

# # Calculate the number of bins
# num_bins = int(total_time / bin_size)

# # Create the histogram
# hist, bin_edges = np.histogram(df[column_to_plot], bins=num_bins, range=(0, total_time))

# # Calculate bin centers for plotting
# bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# # Create the plot
# plt.figure(figsize=(12, 6))

# # Plot the histogram
# plt.bar(bin_centers, hist, width=bin_size, alpha=0.7, color='skyblue', edgecolor='black')

# # Customize the plot
# plt.title(f'Histogram of {column_to_plot}')
# plt.xlabel('Time (seconds)')
# plt.ylabel('Count')
# plt.grid(True, alpha=0.3)

# # Set x-axis limits
# plt.xlim(0, total_time)

# # Add text about bin size
# plt.text(0.98, 0.95, f'Bin size: {bin_size} s', 
#          horizontalalignment='right', verticalalignment='top', 
#          transform=plt.gca().transAxes)

# # Show the plot
# plt.show()


# # File paths
# file_path1 = "/home/abebu/SimQN/security/QuantumClockMitM/reciver/data/received_data_20240715_105026_127.0.0.1.csv"
# file_path2 = "/home/abebu/SimQN/security/QuantumClockMitM/sender/data/arrivals_A.csv"

# # Read the CSV files
# df1 = pd.read_csv(file_path1)
# df2 = pd.read_csv(file_path2)

# # Assuming the first column contains the data we want to compare in both files
# column_to_compare1 = df1.columns[0]
# column_to_compare2 = df2.columns[0]

# # Ensure both dataframes have the same number of rows
# min_rows = min(len(df1), len(df2))
# df1 = df1.head(min_rows)
# df2 = df2.head(min_rows)

# # Compare values
# equal_values = np.isclose(df1[column_to_compare1], df2[column_to_compare2], rtol=1e-5, atol=1e-8)

# # Calculate statistics
# total_values = len(equal_values)
# matching_values = np.sum(equal_values)
# non_matching_values = total_values - matching_values

# print(f"Total values compared: {total_values}")
# print(f"Matching values: {matching_values}")
# print(f"Non-matching values: {non_matching_values}")
# print(f"Percentage of matching values: {(matching_values/total_values)*100:.2f}%")

# # If you want to see the non-matching values:
# if non_matching_values > 0:
#     print("\nNon-matching values (first 10):")
#     non_matching_indices = np.where(~equal_values)[0][:10]
#     for idx in non_matching_indices:
#         print(f"Index {idx}: File1 = {df1[column_to_compare1].iloc[idx]}, File2 = {df2[column_to_compare2].iloc[idx]}")
data=[0.001776357002683731, 0.0018320980005517783, 0.002039893000826566, 0.0032569080467638786, 0.0032584890652106796, 0.00453443010906438, 0.004668125149587963, 0.005105461209967037, 0.005508511293284459, 0.007052105354882354, 0.007103805446557647, 0.007426433554827402, 0.007628735669263533, 0.00780236578935873, 0.007874998900941215, 0.009356407039621508, 0.01035002319207692, 0.010444510369519234, 0.012082656539642243, 0.012113112714175698, 0.01245121390410931, 0.012649400084816671, 0.012888785314731017, 0.013783790516038502, 0.013897094770281839, 0.01406824199914876, 0.014376753255024462, 0.014402455510792651, 0.014588145782476048, 0.015371840052517948, 0.015765425352374764, 0.01583029265386743, 0.016598108967030144, 0.01718988029090657, 0.017891418620792018, 0.018049115959246582, 0.018117703300622783, 0.01877783065478158, 0.019012175028419228, 0.01932920041097212, 0.02038218781835261, 0.02216195122142547, 0.02243408061352609, 0.02325865903075325, 0.023554071457032102, 0.023686726907967904, 0.02423591436935344, 0.024956166815356434, 0.025568057309176842, 0.02643294978211199, 0.02655385428028409, 0.02717163776288473, 0.027746187289248256, 0.02819652779236202, 0.029201344341753754, 0.02964151489067657, 0.02985901744201847, 0.03007473698398649, 0.0301605785520756, 0.03096758514089697, 0.03276522074997952, 0.03296297032241936, 0.03355338893274312, 0.03411105454931544, 0.03413039420267021, 0.03436150782675537, 0.03472644750774116, 0.034842711174389716, 0.03512011482882912, 0.03607067453641136, 0.037062913207166814, 0.03837393890220326, 0.04038009461452976, 0.04112245235663452, 0.04195594508286732, 0.042048873797344004, 0.042127594551626305, 0.04256823831859082, 0.04309324910029278, 0.04328048787624067, 0.04358494167467262, 0.04367566247725878, 0.0439139982875301, 0.04472490514326024, 0.04492208796966949, 0.045358841816772494, 0.045428603669336716, 0.04563603751550948, 0.04599348439107608, 0.04608622426453219, 0.04622328816921972, 0.04819843506175014, 0.04842315996983597, 0.048865628902388214, 0.04899638085307096, 0.04905446678970384, 0.049093487749241334, 0.05060680170795808, 0.05083792867327076, 0.05104560767242093, 0.051660461663041196, 0.0517288976802182, 0.05221092969567557, 0.05341125870433082, 0.05353642774377771, 0.054068573778985676, 0.05488794983482004, 0.05546383690323652, 0.05590731098961388, 0.05622276706528946, 0.05673725618185244, 0.056757156277149924, 0.05685394338585696, 0.05692438252493916, 0.05848423464378353, 0.05863125679115434, 0.05908733696765023, 0.05939487810780693, 0.05943261829618871, 0.060246874480818856, 0.06041407167432742]
print(len(data))