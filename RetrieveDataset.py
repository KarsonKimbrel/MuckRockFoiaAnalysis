import json
import requests
import sys


def downloadDataset():
	print('Updating dataset...')
	saveData(updateJuristictions(), './juristictions.json')
	saveData(updateAgencies(), './agencies.json')
	saveData(updateFoiaRequests(), './foiaRequests.json')
	print('Dataset saved.')
	return
def updateJuristictions():
	return retrieveData('https://www.muckrock.com/api_v1/jurisdiction/')
def updateAgencies():
	return retrieveData('https://www.muckrock.com/api_v1/agency/')
def updateFoiaRequests():
	return retrieveData('https://www.muckrock.com/api_v1/foia/')
def retrieveData(url):
	responses = []
	t = False
	while t or True:
		print(url)
		resp = requests.get(url)
		data = resp.json()
		url = data['next']
		if url == None:
			break
		for result in data['results']:
			responses.append(result)
		t = False
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

