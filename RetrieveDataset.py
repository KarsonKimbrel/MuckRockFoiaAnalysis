import json
import requests
import sys


def downloadDataset():
	print('Updating dataset...')
	#saveData(updateJuristictions(), './juristictions.json')
	#saveData(updateAgencies(), './agencies.json')
	saveData(updateFoiaRequests(), './foiaRequests.json')
	print('Dataset saved.')
	return
def updateJuristictions():
	return retrieveData('https://www.muckrock.com/api_v1/jurisdiction/', None)
def updateAgencies():
	return retrieveData('https://www.muckrock.com/api_v1/agency/', None)
def updateFoiaRequests():
	return retrieveData('https://www.muckrock.com/api_v1/foia/', None)
def retrieveData(baseUrl, params):
	responses = []
	page = 1
	t = False
	while True:
		retryNum = 0
		while True:
			url = baseUrl + '?page=' + str(page)
			if params != None:
				url = url + '&' + params
			try:
				resp = requests.get(url)
				data = resp.json()
				page = page + 1
				print(url)
				break
			except:
				pass
			retryNum = retryNum + 1
			if retryNum == 3:
				retryNum = 0
				page = page + 1
		for result in data['results']:
			responses.append(result)
		url = data['next']
		if url == None:
			break
	return responses


def loadData():
	juristictions = loadDataByName('./juristictions.json')
	agencies = loadDataByName('./agencies.json')
	foiaRequests = loadDataByName('./foiaRequests.json')
	return juristictions, agencies, foiaRequests


def loadDataByName(fileName):
	file = open(fileName, 'r', encoding='utf8')
	data = json.load(file)
	file.close()
	return data


def saveData(data, fileName):
	file = open(fileName, 'w')
	json.dump(data, file)
	file.close()
	return


def column(matrix, i):
    return [row[i] for row in matrix]

