#encoding:utf-8


import json,datetime,requests
import time,os
import sys
sys.path.append("../")
from cocoNLP_my.extractor import extractor
class WeatherQuery():
    def __init__(self):
        self.cityName=None
        self.timeWeather=None
    def weather_query(self,cityName,timeWeather=None):
        # http://wthrcdn.etouch.cn/weather_mini?city=杭州
        weatherUrl = "http://wthrcdn.etouch.cn/weather_mini?city="  #测试用接口
        # weatherUrl = "http://wthrcdnetouch.axnfw.net/weather_mini?city="  #接入nginx权限的发布用地址
        weatherResp = requests.get(weatherUrl + cityName)
        # print(weatherUrl+cityName,weatherResp)
        d = weatherResp.json()
        # print(d)
        if (d['status'] >= 1000):
            weathers_info={}
            for dd in [d['data']['yesterday']]:
                i=dd['date'].index("日")
                weathers_info[dd['date'][:i]]=dd
            for dd in d['data']['forecast']:
                i = dd['date'].index("日")
                weathers_info[dd['date'][:i]] = dd
            # timeWeather = '2019-12-26 00:00:00'
            # print(timeWeather.day,type(timeWeather.day),info)

            # if timeWeather is None:
            #     res_s=''
            #     for info in list(weathers_info.values())[0:]:
            #         if 'fx' in info:
            #             info['fengxiang']=info['fx']
            #         if 'fl' in info:
            #             info['fengli'] = info['fl']
            #         res = " 城市:{}\n 时间:{}\n 天气状况:{}\n 最低温度:{}\n 最高温度:{}\n 风力:{}{}\n ".format(d["data"]["city"],
            #                                                                                  info['date'], info['type'],
            #                                                                                  info['low'], info['high'],
            #                                                                                  info['fengxiang'],
            #                                                                                  info['fengli'])
            #         res_s+=res
            #     return 1,res_s
            if timeWeather is None:
                timeWeather=datetime.datetime.now()
            else:
                timeWeather = datetime.datetime.strptime(timeWeather, '%Y-%m-%d %H:%M:%S')
            day_num = str(timeWeather.day)
            if day_num not in weathers_info.keys():
                res="小诺支持查询当前天，前一天和后四天的天气，查询时间超出了小诺能查询的范围。"
            else:
                info = weathers_info[day_num]
                if 'fx' in info:
                    info['fengxiang'] = info['fx']
                if 'fl' in info:
                    info['fengli'] = info['fl']
                res = " 城市:{}\n 时间:{}\n 天气状况:{}\n 最低温度:{}\n 最高温度:{}\n 风力:{}{}\n ".format(d["data"]["city"],
                                                                                         info['date'], info['type'],
                                                                                         info['low'], info['high'],
                                                                                         info['fengxiang'],                                                                           info['fengli'])
            return 1, res
        else:
            return -1,"无查询结果！"

    def query(self,text = "我想查芜湖今天的天气"):
        '''这里location取市级单位，time取第一个时间点'''
        def get_timeWeather(times):
            # timeWeather = datetime.datetime.now()
            if "timespan" in times.keys():
                timeWeather = times['timespan'][0]
                return timeWeather
            elif 'timestamp'in times.keys():
                timeWeather = times['timestamp']
                return timeWeather
            return None
        def get_cityName(locations):
            if locations:
                dir_path = os.path.dirname(os.path.abspath(__file__))
                print(dir_path)
                f = open(dir_path+'/city.json', 'r',encoding='utf-8')  # 读取json文件
                # f = open('./city.json', 'r',encoding='utf-8')  # 读取json文件
                cities = json.load(f)
                f.close()
                for l in locations:
                    if "市" in l:
                        i = l.index("市")
                        if l[:i] in cities:
                            cityName = l[:i]
                            return cityName
                    else:
                        if l in cities:
                            cityName = l
                            return cityName
                return None
        ex = extractor()
        print("WeatherQuery.query.ex",type(text),text,ex)
        locations = ex.extract_locations(text)
        print(locations)
        # locations="杭州"
        times = json.loads(ex.extract_time(text))
        print("提取时间 & 城市名",locations,times)

        timeWeather = get_timeWeather(times)
        cityName = get_cityName(locations)
        print("text:{},cityName:{},timeWeather:{}".format(text,cityName,timeWeather))
        if cityName is None:
            reslut={"status":-1,"cityName":cityName,"timeWeather":timeWeather,"result":"您好，小诺现在仅支持市级城市天气的查询功能。请重新输入要查询的城市名："}
        if locations and cityName:
            if timeWeather:
                print("查询:{}，{}的天气".format(cityName,timeWeather))
            else:
                print("查询{}天气的全部结果".format(cityName))
            f,res=self.weather_query(cityName, timeWeather)
            if f==1:
                reslut = {"status": 1,"cityName":cityName,"timeWeather":timeWeather, "result": res}
            else:
                reslut = {"status": -1,"cityName":cityName,"timeWeather":timeWeather, "result": res}
            print(reslut)
        return reslut
if __name__ == "__main__":
    wq=WeatherQuery()
    text=""
    while text!='./stop':
        text=input("请输入：")
        wq.query(text)
    # print(wq.weather_query("阜阳"))
    # print(e)
