import datetime as dt
from itertools import filterfalse
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import operator
import RetrieveDataset as retrieve
import scipy.stats as stats


YEAR_START = 2013
YEAR_END = 2017


def main():
	#retrieve.downloadDataset()
	jurisdictions, agencies, foiaRequests = retrieve.loadData()
	printTotals(jurisdictions, agencies, foiaRequests)
	filterData(jurisdictions, agencies, foiaRequests)
	sorted(foiaRequests, key=operator.itemgetter('date_submitted'), reverse=False)
	print('Post Filter:')
	printTotals(jurisdictions, agencies, foiaRequests)
	printRequestStatuses(foiaRequests)
	plotRequestStatuses(foiaRequests)
	plotSuccessesByYear(foiaRequests)
	plotSuccessesByMonth(foiaRequests)
	plotSuccessesByAverageMonth(foiaRequests)
	plt.show()
	return


def filterData(jurisdictions, agencies, foiaRequests):
	foiaRequests[:] = filterfalse(shouldFilterFoiaRequest, foiaRequests)
	return


def shouldFilterFoiaRequest(foiaRequest):
	if foiaRequest['date_submitted'] == None:
		return True
	if getDate(foiaRequest['date_submitted']).year > YEAR_END:
		return True
	if getDate(foiaRequest['date_submitted']).year < YEAR_START:
		return True
	return False


def plotRequestStatuses(foiaRequests):
	statuses = []
	for foia in foiaRequests:
		statuses.append(foia['status'])
		foia['taggedstatus'] = 'test'
	statuses = np.unique(statuses)
	statusCount = {}
	total = 0
	totalSuccessful = 0
	totalUnsuccessful = 0
	for status in statuses:
		statusCount[status] = 0
	for foia in foiaRequests:
		statusCount[foia['status']] += 1
		total += 1
		if isFoiaRequestSuccessful(foia):
			totalSuccessful += 1
		else:
			totalUnsuccessful += 1
	other = 0
	for status in statuses:
		if ((statusCount[status] / total) * 100) < 4.0:
			other += statusCount[status]
			del statusCount[status]
	statusCount['other'] = other
	statusCount = dict(sorted(statusCount.items(), key=operator.itemgetter(1)))
	
	fig = figure(7)
	plt.title('Breakdown of FOIA Request Statuses')
	plt.pie(list(statusCount.values()), autopct='%1.1f%%', counterclock=False)
	plt.axis('equal')
	plt.legend(labels=list(statusCount.keys()))
	plt.savefig('figures/statuses.png')
	
	legend = ['Successful', 'Unsuccessful']
	colors = ['green', 'red']
	fig = figure(7)
	plt.title('Breakdown of FOIA Request Statuses')
	plt.pie([totalSuccessful, totalUnsuccessful], colors=colors, autopct='%1.1f%%', counterclock=True)
	plt.axis('equal')
	plt.legend(labels=legend)
	plt.savefig('figures/statuses_simple.png')
	return


def plotSuccessesByYear(foiaRequests):
	fig = figure()
	plt.title('FOIA Requests By Year')
	plt.xlabel('Year')
	plt.ylabel('Number of Requests')
	years = {}
	for foia in foiaRequests:
		date = getDate(foia['date_submitted'])
		year = date.year
		if year not in years.keys():
			years[year] = 0
		years[year] += 1
	X = years.keys()
	Y = years.values()
	plt.bar(X, Y)
	plt.savefig('figures/yearly_total.png')
	
	fig = figure()
	plt.title('FOIA Requests By Year')
	plt.xlabel('Year')
	plt.ylabel('Number of Requests')
	foiaRequestsSuccess = filterfalse(isFoiaRequestFailed, foiaRequests)
	for year in years:
		year = 0
	years = sorted(years)
	successYears = {}
	for year in years:
		successYears[year] = 0
	for foia in foiaRequestsSuccess:
		date = getDate(foia['date_submitted'])
		year = date.year
		if year not in successYears.keys():
			successYears[year] = 0
		successYears[year] += 1
	failedYears = {}
	for year in years:
		failedYears[year] = 0
	foiaRequestsFailed =  filterfalse(isFoiaRequestSuccessful, foiaRequests)
	for foia in foiaRequestsFailed:
		date = getDate(foia['date_submitted'])
		year = date.year
		if year not in failedYears.keys():
			failedYears[year] = 0
		failedYears[year] += 1
	Xsuccess = np.array(list(successYears.keys()))
	Ysuccess = np.array(list(successYears.values()))
	Xfailed = np.array(list(failedYears.keys()))
	Yfailed = np.array(list(failedYears.values()))
	sets = [
		(Xsuccess, Ysuccess, 'Successful', 'green'),
		(Xfailed, Yfailed, 'Unsuccessful', 'red')]
	plotMultiBarGraph(sets)
	plt.legend()
	plt.savefig('figures/yearly_breakdown.png')
	
	fig = figure()
	plt.title('Percentage of Successful FOIA Requests By Year')
	plt.xlabel('Year')
	plt.ylabel('Percent of Successful Requests')
	X = years
	Y = (Ysuccess / (Ysuccess + Yfailed)) * 100
	plt.bar(years, Y)
	plt.savefig('figures/yearly_percent_successful.png')
	
	return


def plotSuccessesByMonth(foiaRequests):
	dates = {}
	minDate = None
	maxDate = None
	Xlabels = []
	for year in np.arange(YEAR_START-1, YEAR_END+2):
		for month in np.arange(1, 13):
			if (month < 10):
				key = str(year) + '-0' + str(month)
			else:
				key = str(year) + '-' + str(month)
			Xlabels.append(key)
			dates[key] = {}
			dates[key]['total'] = 0
			dates[key]['successes'] = 0
			dates[key]['failures'] = 0
	for foia in foiaRequests:
		date = getDate(foia['date_submitted'])
		year = date.year
		if (date.month < 10):
			key = str(date.year) + '-0' + str(date.month)
		else:
			key = str(date.year) + '-' + str(date.month)
		dates[key]['total'] += 1
		if isFoiaRequestSuccessful(foia):
			dates[key]['successes'] += 1
		else:
			dates[key]['failures'] += 1
	Ytotal = []
	Ysuccesses = []
	Yfailures = []
	YsuccessPercentage = []
	for y in dates.values():
		Ytotal.append(y['total'])
		Ysuccesses.append(y['successes'])
		Yfailures.append(y['failures'])
		if y['total'] != 0:
			YsuccessPercentage.append(y['successes'] / y['total'] * 100)
		else:
			YsuccessPercentage.append(0)
	
	fig = figure()
	plt.title('FOIA Requests By Year and Month')
	plt.xlabel('Month')
	plt.ylabel('Number of Requests')
	plt.bar(Xlabels, Ytotal)
	plt.xticks(Xlabels, Xlabels, rotation='vertical')
	plt.xlim(str(YEAR_START-1) + '-12', str(YEAR_END+1) + '-01')
	plt.savefig('figures/monthly_total.png')
	
	fig = figure()
	plt.title('FOIA Requests By Year and Month')
	plt.xlabel('Month')
	plt.ylabel('Number of Requests')
	X = np.arange(0, len(Xlabels))
	sets = [
		(X, Ysuccesses, 'Successful', 'green'),
		(X, Yfailures, 'Unsuccessful', 'red')]
	plotMultiBarGraph(sets)
	plt.xticks(X, Xlabels, rotation='vertical')
	xRange = [-1,-1]
	i = 0
	for label in Xlabels:
		if label == str(YEAR_START-1) + '-12':
			xRange[0] = i
		if label == str(YEAR_END+1) + '-01':
			xRange[1] = i
		i += 1
	plt.xlim(xRange[0], xRange[1])
	plt.legend()
	plt.savefig('figures/monthly_breakdown.png')
	
	fig = figure()
	plt.title('Percentage of Successful FOIA Requests By Year and Month')
	plt.xlabel('Month')
	plt.ylabel('Percentage of Successful Requests')
	plt.bar(Xlabels, YsuccessPercentage)
	plt.xticks(Xlabels, Xlabels, rotation='vertical')
	plt.xlim(str(YEAR_START-1) + '-12', str(YEAR_END+1) + '-01')
	plt.savefig('figures/monthly_percent_successful.png')
	
	minIndex = None
	maxIndex = None
	x = 0
	for date in Xlabels:
		if Ytotal[x] != 0:
			if minIndex == None and Ytotal[x] != 0:
				minIndex = x
			maxIndex = x
		x += 1
	x = 0
	yFilteredTotal = []
	yFilteredSuccessPercent = []
	for date in Xlabels:
		if x >= minIndex and x <= maxIndex:
			yFilteredTotal.append(Ytotal[x])
			yFilteredSuccessPercent.append(YsuccessPercentage[x])
		x += 1
		
	fig = figure()
	plt.title('Histogram of FOIA Requests By Year and Month')
	n, bins, patches = plt.hist(yFilteredTotal, bins='auto', edgecolor='black', normed=True)
	mu = np.mean(yFilteredTotal)
	sigma = np.std(yFilteredTotal)
	#x = np.linspace(bins[0], bins[len(bins)-1], 100)
	x = np.linspace(0, bins[len(bins)-1], 100)
	fit = stats.norm.pdf(x, mu, sigma)
	plt.plot(x, fit, 'r--')
	plt.savefig('figures/monthly_histogram.png')
		
	fig = figure()
	plt.title('Histogram of Percentages of Successful FOIA Requests By Year and Month')
	n, bins, patches = plt.hist(yFilteredSuccessPercent, bins='auto', edgecolor='black', normed=True)
	mu = np.mean(yFilteredSuccessPercent)
	sigma = np.std(yFilteredSuccessPercent)
	x = np.linspace(0, 100, 100)
	fit = stats.norm.pdf(x, mu, sigma)
	plt.plot(x, fit, 'r--')
	plt.savefig('figures/monthly_histogram_successful_percentage.png')
	
	normalityTests(yFilteredTotal, 'FOIA Requests By Year and Month', 0.01)
	normalityTests(yFilteredSuccessPercent, 'Percentages of Successful FOIA Requests By Year and Month', 0.01)
	return

	
def normalityTests(data, title, significanceLevel=0.05):
	# Shapiro-Wilk test for normality
	W, pValue = stats.shapiro(data)
	print('Shapiro-Wilk Test For Normality')
	print('H0: The distribution of ' + title + ' is normal.')
	print('Ha: The distribution of ' + title + ' is not normal.')
	print('TS: W=' + '{:.4f}'.format(W))
	if pValue <= significanceLevel:
		print('P-Value: P=' + '{:.4f}'.format(pValue) + ' < ' + '{:.4f}'.format(significanceLevel))
		print('Reject Ho. At the alpha={:.2f}'.format(significanceLevel) + ' significance level, there is sufficiant evidence to indicate that the distribution of ' + title + ' is not normal.')
		print('')
		print('The distribution of ' + title + ' IS NOT normal.')
	else:
		print('P-Value: P=' + '{:.4f}'.format(pValue) + ' !< ' + '{:.4f}'.format(significanceLevel))
		print('Do not reject Ho. At the alpha={:.2f}'.format(significanceLevel) + ' significance level, there is insufficiant evidence to indicate that the distribution of ' + title + ' is not normal.')
		print('')
		print('The distribution of ' + title + ' IS normal.')
	print('')
	print('')
	
	# Anderson-Darling test for normality
	A2, critialValues, significanceLevels = stats.anderson(data, dist='norm')
	print('Anderson-Darling Test For Normality')
	print('H0: The distribution of ' + title + ' is normal.')
	print('Ha: The distribution of ' + title + ' is not normal.')
	critialValues = dict(zip(significanceLevels, critialValues))
	criticalValue = critialValues[significanceLevel*100]
	if A2 < criticalValue:
		print('TS: A2=' + '{:.4f}'.format(A2) + ' < ' + '{:.4f}'.format(criticalValue))
		print('Do not reject Ho. At the alpha={:.2f}'.format(significanceLevel) + ' significance level, there is insufficiant evidence to indicate that the distribution of ' + title + ' is not normal.')
		print('')
		print('The distribution of ' + title + ' IS normal.')
	else:
		print('TS: A2=' + '{:.4f}'.format(A2) + ' !< ' + '{:.4f}'.format(criticalValue))
		print('Reject Ho. At the alpha={:.2f}'.format(significanceLevel) + ' significance level, there is sufficiant evidence to indicate that the distribution of ' + title + ' is not normal.')
		print('')
		print('The distribution of ' + title + ' IS NOT normal.')
	print('')
	print('')
	return
	

def plotSuccessesByAverageMonth(foiaRequests):
	months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
	X = np.arange(0, 12)
	data = {}
	data['unnormalized'] = {}
	data['normalized'] = {}
	data['months'] = {}
	data['years'] = {}
	for year in np.arange(YEAR_START-1, YEAR_END+1):
		data['years'][year] = {}
	for month in X:
		data['unnormalized'][month] = {}
		data['unnormalized'][month]['total'] = 0
		data['unnormalized'][month]['successes'] = 0
		data['unnormalized'][month]['failures'] = 0
		data['normalized'][month] = {}
		data['normalized'][month]['total'] = 0
		data['normalized'][month]['successes'] = 0
		data['normalized'][month]['failures'] = 0
		data['normalized'][month]['percentageSuccess'] = 0
		for year in np.arange(YEAR_START-1, YEAR_END+1):
			data['years'][year][month] = {}
			data['years'][year][month]['total'] = 0
			data['years'][year][month]['successes'] = 0
			data['years'][year][month]['failures'] = 0
	for foia in foiaRequests:
		date = getDate(foia['date_submitted'])
		year = date.year
		month = date.month-1
		data['unnormalized'][month]['total'] += 1
		data['years'][year][month]['total'] += 1
		if isFoiaRequestSuccessful(foia):
			data['unnormalized'][month]['successes'] += 1
			data['years'][year][month]['successes'] += 1
		else:
			data['unnormalized'][month]['failures'] += 1
			data['years'][year][month]['failures'] += 1
	for month in X:
		activeYears = 0
		sumTotal = 0
		sumSuccesses = 0
		sumfailures = 0
		for year in np.arange(YEAR_START-1, YEAR_END+1):
			if data['years'][year][month]['total'] != 0:
				sumTotal = data['years'][year][month]['total']
				sumSuccesses = data['years'][year][month]['successes']
				sumfailures = data['years'][year][month]['failures']
				activeYears += 1
		data['normalized'][month]['total'] = sumTotal / activeYears
		data['normalized'][month]['successes'] = sumSuccesses / activeYears
		data['normalized'][month]['failures'] = sumfailures / activeYears
		data['normalized'][month]['percentageSuccess'] = (sumSuccesses / activeYears) / (sumTotal / activeYears) * 100
	unnormalizedTotal = []
	unnormalizedSuccesses = []
	unnormalizedFailures = []
	unnormalizedPercentageSuccess = []
	normalizedTotal = []
	normalizedSuccesses = []
	normalizedFailures = []
	normalizedPercentageSuccess = []
	for month in X:
		unnormalizedTotal.append(data['unnormalized'][month]['total'])
		unnormalizedSuccesses.append(data['unnormalized'][month]['successes'])
		unnormalizedFailures.append(data['unnormalized'][month]['failures'])
		unnormalizedPercentageSuccess.append(data['unnormalized'][month]['successes'] / data['unnormalized'][month]['total'] * 100)
		normalizedTotal.append(data['normalized'][month]['total'])
		normalizedSuccesses.append(data['normalized'][month]['successes'])
		normalizedFailures.append(data['normalized'][month]['failures'])
		normalizedPercentageSuccess.append(data['normalized'][month]['percentageSuccess'])
	
	fig = figure()
	plt.title('FOIA Requests By Month (Unnormalized)')
	plt.xlabel('Month')
	plt.ylabel('Number of Requests')
	plt.bar(X, unnormalizedTotal)
	plt.xticks(X, months, rotation='horizontal')
	plt.savefig('figures/month_unnormalized_total.png')
	
	fig = figure()
	plt.title('FOIA Requests By Month (Unnormalized)')
	plt.xlabel('Month')
	plt.ylabel('Number of Requests')
	sets = [
		(X, unnormalizedSuccesses, 'Successful', 'green'),
		(X, unnormalizedFailures, 'Unsuccessful', 'red')]
	plotMultiBarGraph(sets)
	plt.xticks(X, months, rotation='horizontal')
	plt.savefig('figures/month_unnormalized_breakdown.png')
	
	fig = figure()
	plt.title('Percentage of Successful FOIA Requests By Month (Unnormalized)')
	plt.xlabel('Month')
	plt.ylabel('Percentage of Successful Requests')
	plt.bar(X, unnormalizedPercentageSuccess)
	plt.xticks(X, months, rotation='horizontal')
	plt.savefig('figures/month_unnormalized_percent_successful.png')
	
	fig = figure()
	plt.title('FOIA Requests By Month (Normalized)')
	plt.xlabel('Month')
	plt.ylabel('Average Number of Requests')
	plt.bar(X, normalizedTotal)
	plt.xticks(X, months, rotation='horizontal')
	plt.savefig('figures/month_normalized_total.png')
	
	fig = figure()
	plt.title('FOIA Requests By Month (Normalized)')
	plt.xlabel('Month')
	plt.ylabel('Average Number of Requests')
	sets = [
		(X, normalizedSuccesses, 'Successful', 'green'),
		(X, normalizedFailures, 'Unsuccessful', 'red')]
	plotMultiBarGraph(sets)
	plt.xticks(X, months, rotation='horizontal')
	plt.savefig('figures/month_normalized_breakdown.png')
	
	fig = figure()
	plt.title('Percentage of Successful FOIA Requests By Month (Normalized)')
	plt.xlabel('Month')
	plt.ylabel('Average Percentage of Successful Requests')
	plt.bar(X, normalizedPercentageSuccess)
	plt.xticks(X, months, rotation='horizontal')
	plt.savefig('figures/month_normalized_percent_successful.png')
	
	return


def printRequestStatuses(foiaRequests):
	print('Record Statuses:')
	statuses = []
	for foia in foiaRequests:
		statuses.append(foia['status'])
		foia['taggedstatus'] = 'test'
	statuses = np.unique(statuses)
	statusCount = {}
	for status in statuses:
		statusCount[status] = 0
	for foia in foiaRequests:
		statusCount[foia['status']] += 1
	for status in statuses:
		print('{:12}'.format(status) + str(statusCount[status]))
	print('')
	return


def printTotals(jurisdictions, agencies, foiaRequests):
	print('Totals:')
	print('{:16}'.format('Jurisdictions:') + str(len(jurisdictions)))
	print('{:16}'.format('Agencies:') + str(len(agencies)))
	print('{:16}'.format('FoiaRequests:') + str(len(foiaRequests)))
	print('')
	return


def getDate(dateStr):
	return dt.datetime.strptime(dateStr, "%Y-%m-%d").date()


def isFoiaRequestSuccessful(foiaRequest):
	status = foiaRequest['status']
	if status == 'done':
		return True
	return False


def isFoiaRequestFailed(foiaRequest):
	return not isFoiaRequestSuccessful(foiaRequest)


def plotMultiBarGraph(sets):
	numCategories = len(sets)
	w = 0.8 / numCategories
	ind = np.arange(len(sets[0][0]))
	t = 0
	for set in sets:
		dataX = set[0]
		dataY = set[1]
		label = set[2]
		color = set[3]
		plt.bar(ind + t, dataY, width=w, color=color, label=label)
		t += w
	plt.xticks(ind + w / numCategories, tuple(sets[0][0]))
	plt.legend()
	return


def figure(w=10, h=5):
	fig = plt.figure(figsize=(w, h))
	fig.subplots_adjust(bottom=0.2)
	return fig


main()