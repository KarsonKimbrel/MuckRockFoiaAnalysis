from collections import defaultdict
import datetime as dt
from functools import lru_cache
from itertools import filterfalse
from itertools import groupby
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import operator
import plotly.express as px
import plotly.graph_objs as go
import RetrieveDataset as retrieve
import scipy.stats as stats

YEAR_START = 2014
YEAR_END = 2018
DATE_START = '2014-01-01'
DATE_END = '2019-01-01'
DISP_FULL_TESTS = False


def main():
	#retrieve.downloadDataset()
	jurisdictions, agencies, foiaRequests = retrieve.loadData()
	print()
	printTotals(jurisdictions, agencies, foiaRequests)
	filterData(jurisdictions, agencies, foiaRequests)
	print('Sorting data...')
	foiaRequests = sorted(foiaRequests, key=operator.itemgetter('datetime_submitted'), reverse=False)
	annotateData(jurisdictions, agencies, foiaRequests)
	print()
	print('Post Filter:')
	printTotals(jurisdictions, agencies, foiaRequests)
	
	printRequestStatuses(foiaRequests)
	plotRequestStatuses(foiaRequests)
	plotSuccessesByDay(foiaRequests)
	plotSuccessesByYearMonth(foiaRequests)
	plotSuccessesByYear(foiaRequests)
	plotSuccessesByMonth(foiaRequests)
	plt.show()
	#printDocumentTitlesOnDay(foiaRequests, dt.datetime(year=2013, month=3, day=19))
	
	return


def filterData(jurisdictions, agencies, foiaRequests):
	print('Filtering data...')
	foiaRequests[:] = filterfalse(shouldFilterFoiaRequest, foiaRequests)
	return


def annotateData(jurisdictions, agencies, foiaRequests):
	print('Annotating data...')
	for request in foiaRequests:
		date = getDate(request['datetime_submitted'])
		request['date_day'] = date.day
		request['date_month'] = date.month
		request['date_year'] = date.year
		request['date_year_month'] = date.strftime('%Y-%m')
		request['date_year_month_day'] = date.strftime('%Y-%m-%d')
		request['num_communications'] = len(request['communications'])
		request['is_successful'] = request['status'] == 'done' or request['status'] == 'partial'
	return


def shouldFilterFoiaRequest(foiaRequest):
	if foiaRequest['datetime_submitted'] == None:
		return True
	if getDate(foiaRequest['datetime_submitted']).year > YEAR_END:
		return True
	if getDate(foiaRequest['datetime_submitted']).year < YEAR_START:
		return True	
	if 'Library Complaint and Censorship Records' in foiaRequest['title']:
		return True
	if foiaRequest['status'] == 'processed' or foiaRequest['status'] == 'ack':
		return True
	#if foiaRequest['agency'] != 10:
		#return True
	return False


def plotRequestStatuses(foiaRequests):
	requests = sorted(foiaRequests, key=lambda x: x['status'])
	totalRequests = len(requests)
	statuses = defaultdict(int)
	simpleStatuses = defaultdict(int)
	groups = groupby(requests, lambda x: x['status'])
	for status, requests in groups:
		numRequests = len(list(requests))
		if numRequests / float(totalRequests) >= 0.04:
			statuses[status] = numRequests
		else:
			statuses['other'] += numRequests
		simpleStatuses[isStatusSuccessful(status)] += numRequests
			
	statusCount = dict(sorted(statuses.items(), key=operator.itemgetter(1)))
	
	fig = figure()
	plt.title('Breakdown of FOIA Request Statuses')
	plt.pie(list(statusCount.values()), autopct='%1.1f%%', counterclock=False)
	plt.axis('equal')
	plt.legend(labels=list(statusCount.keys()))
	plt.savefig('figures/statuses.png')
	
	legend = ['Successful', 'Unsuccessful']
	colors = ['green', 'red']
	fig = figure()
	plt.title('Breakdown of FOIA Request Statuses')
	plt.pie([simpleStatuses[True], simpleStatuses[False]], colors=colors, autopct='%1.1f%%', counterclock=True)
	plt.axis('equal')
	plt.legend(labels=legend)
	plt.savefig('figures/statuses_simple.png')
	
	'''
	df = list(map(lambda k, v: {'status': k, 'count': v}, statusCount.keys(), statusCount.values()))
	fig = px.pie(df, names='status', values='count')
	fig.show()
	
	df = list(map(lambda k, v: {'status': k, 'count': v}, legend, [totalSuccessful, totalUnsuccessful]))
	fig = px.pie(df, names='status', values='count')
	fig.show()
	'''
	return


def plotSuccessesByGrouping(foiaRequests, dateRange, groupingKey, groupingName, groupingFilePrefix, plotHistograms=True, plotBargraphs=True, doNormalityTests=True):
	requestsGrouped = groupby(foiaRequests, key=lambda x: x[groupingKey])
	rangeTotal = defaultdict(int)
	rangeSuccesses = defaultdict(int)
	rangeFailures = defaultdict(int)
	rangeSuccessPercent = defaultdict(float)
	for dateYear, items in requestsGrouped:
		requests = list(items)
		numRequests = len(requests)
		rangeTotal[dateYear] += numRequests
		for request in requests:
			if request['is_successful']:
				rangeSuccesses[dateYear] += 1
			else:
				rangeFailures[dateYear] += 1
		rangeSuccessPercent[dateYear] = (float(rangeSuccesses[dateYear]) / float(numRequests)) * 100
	
	if plotBargraphs:
		dateRangeEnum = list(zip(*enumerate(dateRange)))
		
		fig = figure()
		plt.title('FOIA Requests By ' + groupingName)
		plt.xlabel(groupingName)
		plt.ylabel('Number of Requests')
		plt.bar(dateRangeEnum[1], rangeTotal.values())
		if len(dateRange) > 12:
			plt.xticks(dateRangeEnum[0], dateRangeEnum[1], rotation='vertical')
		plt.savefig('figures/' + groupingFilePrefix + '/total.png')
		
		fig = figure()
		plt.title('FOIA Requests By ' + groupingName)
		plt.xlabel(groupingName)
		plt.ylabel('Number of Requests')
		sets = [
			(dateRangeEnum[1], [rangeSuccesses[date] for date in dateRange], 'Successful', 'green'),
			(dateRangeEnum[1], [ rangeFailures[date] for date in dateRange], 'Unsuccessful', 'red')]
		plotMultiBarGraph(sets)
		if len(dateRange) > 12:
			plt.xticks(dateRangeEnum[0], dateRangeEnum[1], rotation='vertical')
		plt.legend()
		plt.savefig('figures/' + groupingFilePrefix + '/total_breakdown.png')
		
		fig = figure()
		plt.title('Percentage of Successful FOIA Requests By ' + groupingName)
		plt.xlabel(groupingName)
		plt.ylabel('Percent of Successful Requests')
		plt.bar(dateRangeEnum[1], [rangeSuccessPercent[date] for date in dateRange])
		if len(dateRange) > 12:
			plt.xticks(dateRangeEnum[0], dateRangeEnum[1], rotation='vertical')
		plt.savefig('figures/' + groupingFilePrefix + '/percent_successful.png')
	
	if plotHistograms:
		fig = figure()
		plt.title('Histogram of FOIA Requests By ' + groupingName)
		yFilteredTotal = list(rangeTotal.values())
		n, bins, patches = plt.hist(yFilteredTotal, bins='auto', edgecolor='black', density=True)
		mu = np.mean(yFilteredTotal)
		sigma = np.std(yFilteredTotal)
		x = np.linspace(0, bins[len(bins)-1], 100)
		fit = stats.norm.pdf(x, mu, sigma)
		plt.plot(x, fit, 'r--')
		plt.savefig('figures/' + groupingFilePrefix + '/histogram.png')
		
		fig = figure()
		plt.title('Histogram of Percentages of Successful FOIA Requests By ' + groupingName)
		yFilteredSuccessPercent = list(rangeSuccessPercent.values())
		n, bins, patches = plt.hist(yFilteredSuccessPercent, bins='auto', edgecolor='black', density=True)
		mu = np.mean(yFilteredSuccessPercent)
		sigma = np.std(yFilteredSuccessPercent)
		x = np.linspace(0, 100, 100)
		fit = stats.norm.pdf(x, mu, sigma)
		plt.plot(x, fit, 'r--')
		plt.savefig('figures/' + groupingFilePrefix + '/histogram_successful_percentage.png')
	
	if doNormalityTests:
		normalityTests(yFilteredTotal, 'FOIA Requests By ' + groupingName, 0.01)
		normalityTests(yFilteredSuccessPercent, 'Percentages of Successful FOIA Requests By ' + groupingName, 0.01)
	
	return


def plotSuccessesByDay(foiaRequests):
	return plotSuccessesByGrouping(foiaRequests, getYearMonthDayDateRange(), 'date_year_month_day', 'Day', 'daily', plotBargraphs=False)


def plotSuccessesByYearMonth(foiaRequests):
	return plotSuccessesByGrouping(foiaRequests, getYearMonthDateRange(), 'date_year_month', 'Year-Month', 'year_monthly')


def plotSuccessesByMonth(foiaRequests):
	resortedRequests = sorted(foiaRequests, key=lambda x: x['date_month'])
	return plotSuccessesByGrouping(foiaRequests, getMonthDateRange(), 'date_month', 'Month', 'month', plotHistograms=False, doNormalityTests=False)


def plotSuccessesByYear(foiaRequests):
	return plotSuccessesByGrouping(foiaRequests, getYearDateRange(), 'date_year', 'Year', 'yearly', plotHistograms=False, doNormalityTests=False)


def normalityTests(data, title, significanceLevel=0.05):
	# Shapiro-Wilk test for normality
	W, pValue = stats.shapiro(data)
	if DISP_FULL_TESTS:
		print('Shapiro-Wilk Test For Normality')
		print('H0: The distribution of ' + title + ' is normal.')
		print('Ha: The distribution of ' + title + ' is not normal.')
		print('TS: W=' + '{:.4f}'.format(W))
	if pValue <= significanceLevel:
		if DISP_FULL_TESTS:
			print('P-Value: P=' + '{:.4f}'.format(pValue) + ' < ' + '{:.4f}'.format(significanceLevel))
			print('Reject Ho. At the alpha={:.2f}'.format(significanceLevel) + ' significance level, there is sufficiant evidence to indicate that the distribution of ' + title + ' is not normal.')
			print('')
		print('The distribution of ' + title + ' IS NOT normal.')
	else:
		if DISP_FULL_TESTS:
			print('P-Value: P=' + '{:.4f}'.format(pValue) + ' !< ' + '{:.4f}'.format(significanceLevel))
			print('Do not reject Ho. At the alpha={:.2f}'.format(significanceLevel) + ' significance level, there is insufficiant evidence to indicate that the distribution of ' + title + ' is not normal.')
			print('')
		print('The distribution of ' + title + ' IS normal.')
	if DISP_FULL_TESTS:
		print('')
		print('')
	
	# Anderson-Darling test for normality
	A2, critialValues, significanceLevels = stats.anderson(data, dist='norm')
	
	if DISP_FULL_TESTS:
		print('Anderson-Darling Test For Normality')
		print('H0: The distribution of ' + title + ' is normal.')
		print('Ha: The distribution of ' + title + ' is not normal.')
	critialValues = dict(zip(significanceLevels, critialValues))
	criticalValue = critialValues[significanceLevel*100]
	if A2 < criticalValue:
		if DISP_FULL_TESTS:
			print('TS: A2=' + '{:.4f}'.format(A2) + ' < ' + '{:.4f}'.format(criticalValue))
			print('Do not reject Ho. At the alpha={:.2f}'.format(significanceLevel) + ' significance level, there is insufficiant evidence to indicate that the distribution of ' + title + ' is not normal.')
			print('')
		print('The distribution of ' + title + ' IS normal.')
	else:
		if DISP_FULL_TESTS:
			print('TS: A2=' + '{:.4f}'.format(A2) + ' !< ' + '{:.4f}'.format(criticalValue))
			print('Reject Ho. At the alpha={:.2f}'.format(significanceLevel) + ' significance level, there is sufficiant evidence to indicate that the distribution of ' + title + ' is not normal.')
			print('')
		print('The distribution of ' + title + ' IS NOT normal.')
	if DISP_FULL_TESTS:
		print('')
		print('')
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


def printDocumentTitlesOnDay(foiaRequests, dateTime):
	print('')
	for foia in foiaRequests:
		foiaDate = getDate(foia['datetime_submitted'])
		if dateTime.year == foiaDate.year and dateTime.month == foiaDate.month and dateTime.day == foiaDate.day:
			print('{:16}'.format(foia['status']) + foia['title'])
	return


def getDateStr(date):
	return date.strftime("%Y-%m-%d")


def getDate(dateStr):
	try:
		return dt.datetime.strptime(dateStr, "%Y-%m-%dT%H:%M:%S.%f").date()
	except:
		return dt.datetime.strptime(dateStr, "%Y-%m-%dT%H:%M:%S").date()


@lru_cache(maxsize=None)
def getYearMonthDayDateRange(start=DATE_START, end=DATE_END):
	dateStart = dt.datetime.strptime(start, "%Y-%m-%d")
	dateEnd = dt.datetime.strptime(end, "%Y-%m-%d")
	return [getDateStr(dateStart + dt.timedelta(days=x)) for x in range(0, (dateEnd - dateStart).days)]


@lru_cache(maxsize=None)
def getYearMonthDateRange():
	return [dt.datetime(year=year, month=month, day=1).strftime("%Y-%m") for year in range(YEAR_START, YEAR_END+1) for month in range(1, 13)]


@lru_cache(maxsize=None)
def getMonthDateRange():
	return ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


@lru_cache(maxsize=None)
def getYearDateRange():
	return [year for year in range(YEAR_START, YEAR_END+1)]


def isStatusSuccessful(status):
	return status == 'done'


def isFoiaRequestSuccessful(foiaRequest):
	return isStatusSuccessful(foiaRequest['status'])


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