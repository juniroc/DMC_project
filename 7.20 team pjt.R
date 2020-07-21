# 파일 위치 정의
path <- 'C:\\Users\\LENOVO\\Desktop\\팀플 2차\\Employee_attribute.csv'

# 필요한 패키지 설치
install.packages('pcr')
install.packages('pls')
install.packages('caret')
install.packages('nnet')
install.packages('neuralnet')
install.packages('rJava')
install.packages('FSelector')

# 필요 라이브러리 호출
library(dplyr)
library(car)
library(pcr)
library(pls)
library(caret)
library(correlation)
library(nnet)
library(neuralnet)

# rJava, FSelector 소스 불러오기
source("https://install-github.me/talgalili/installr")
installr::install.java()
library(rJava)
library(FSelector)


# 데이터 전처리작업 
rawdata <- read.csv(path, stringsAsFactors = T)


# 명목형 데이터 numeric 처리 
rawdata$Attrition <- as.numeric(rawdata$Attrition)
rawdata$BusinessTravel <- as.numeric(rawdata$BusinessTravel)
rawdata$Department <- as.numeric(rawdata$Department)
rawdata$EducationField <- as.numeric(rawdata$EducationField)
rawdata$EnvironmentSatisfaction <- as.numeric(rawdata$EnvironmentSatisfaction )
rawdata$Gender <- as.numeric(rawdata$Gender)
rawdata$JobRole <- as.numeric(rawdata$JobRole)
rawdata$MaritalStatus <- as.numeric(rawdata$MaritalStatus                                               )
rawdata$Over18 <- as.numeric(rawdata$Over18)
rawdata$OverTime <- as.numeric(rawdata$OverTime)

# 데이터 구조 확인
str(rawdata)


# 대략적으로 파악을 위해 모든 변수를 이용한 회귀분석을 실행
m <- lm(rawdata$MonthlyIncome~., rawdata)
summary(m)

## 결과
## Department, DistanceFromHome, 
## JobLevel, JobRole, TotalWorkingYear,
## YearsWithCurrManager 가 유의한 변수라 결과나옴

# 그 유의한 변수들로 데이터 프레임 재생성
new_df <- rawdata %>% select(MonthlyIncome, 
                             Department,
                             DistanceFromHome,
                             JobLevel,
                             JobRole,
                             TotalWorkingYears,
                             YearsWithCurrManager
)


# 다시 그 유의한 변수들로 회귀 모델만들기
m1_new <- lm(MonthlyIncome~.,new_df)
summary(m1_new)


## 이것만으로 부족하기에

##############################



# 1) step function을 이용한 변수 소거
step_f <- step(m1_new, direction = 'both')
formula(step_f)

## 결과 
## Department, DistanceFromHome, JobLevel,
## JobRole, TotalWorkingYears, 
## YearsWithCurrManager 가 유의한 설명변수로 판단
## 그러나 AIC 지수가 높아 모델 적합 신뢰도가 
## 높지 않아 추가적인 검정을 더 하였다.


##############################

# 2-1) pcr 주성분분석으로 설명변수의 다중공선성 제거
# 하고 남은 변수들을 이용하고자 함.

# pcr 주성분분석 모델을 만듦.
pcr_out <- pcr(MonthlyIncome~.,data = rawdata,validation='LOO', jackknife=T)
summary(pcr_out)

## 확인 결과
## 성분 누적 결과 설명도 90% 넘는 컴포넌트 == 25개

# 그 25개 성분으로 원래 회귀변수로 모델을 만들어봄

jack.test(pcr_out, ncomp=25)

## 결과
## age, Department, DistanceFromHome, 
## JobLevel, JobRole, TotalworkingYears, 
## YearswithCurrManager 가 유의한 설명변수로 검증

# 2-2) 위에서 공통적으로 유의하다 판단된 변수만으로
# 다시 주성분 분석을하고 그 결과로 잔차를 확인
pcr_out2 <- pcr(MonthlyIncome~., data = new_df, vailidation='LOO', jackknife=T)
summary(pcr_out2)
plot(scale(pcr_out2$residuals[,,3]),main='Residual', xlab = 'Index', ylab = 'Residual')

## 결과
## 잔차들이 골고루 퍼져있기에 ㄱㅊ은 모델인 것으로 판단.

##############################


# 3) 설명변수간 상관관계(상관계수)가 
#  큰 것들을 제거하는 방법으로검증함 (PCA기법)

# 종속변수를 제거한 새로운 데이터프레임 만들고
new_df2 <- new_df[,-1]

# 상관관계가 존재하는 변수를 도출해봤는데
cor_col <- findCorrelation( cor( new_df2 ))
length(cor_col)

## 결과
## 0 으로 서로 상관관계가 큰 독립변수들이 존재하지 않음을 알 수 있었다.


##############################

# 4) 이번에는 설명변수를 연속형변수와 명목형변수인것을 나누어서
# 종속변수와의 상관관계 정도를 확인하는 것으로 검증하였다. 

# 4-1) linear.correlation function으로
#  먼저 연속형 설명 변수들 검증

# linear attribution 데이터 정의
la <- rawdata[,c(1,6,10,13,19,20,24,29,30,32,33,34,35)]

attr_importance_1 <- linear.correlation(MonthlyIncome~., data=la)
attr_importance_1 <- attr_importance_1 %>% arrange(desc(attr_importance))
head(attr_importance_1)

## 결과를 내림차순으로 정렬해보면
## TotalWorkingyears, YearsatCompany 순으로 
## 높은 중요도를 가지는 것을 확인



# 4-2) rank.correlation function으로
# 명목형인 설명 변수들 검증

# MonthlyIncome을 포함한 non_linear attribution 데이터 정의
nla <- rawdata[,-c(1,6,10,13,20,24,29,30,32,33,34,35)]

attr_importance_2 <- rank.correlation(MonthlyIncome~Department+JobLevel+JobRole,data = nla)
attr_importance_2 <- attr_importance_2 %>% arrange(desc(attr_importance))
attr_importance_2

## 결과
## JobLevel이 높은 중요도를 가지는 것을 확인

#####################


# 종합적으로 유의하다 판단된 데이터만으로 새데이터프레임 정의

new_data <- rawdata %>% select(MonthlyIncome, 
                               Department, 
                               DistanceFromHome,
                               JobLevel,
                               TotalWorkingYears,
                               YearsWithCurrManager)


## Department, DistanceFromHome, JobLevel, 
## TotalworkingYears, YearswithCurrManager

#####################

# 5) 선택된 변수들로 회귀분석 실행
f_m <- lm(MonthlyIncome~., data = new_data)
summary(f_m)

## 결과
## 전체적으로 p-value 가 낮게나오며
## R^2 값 또한 0.9 대로 높은 신뢰도를 가짐

# 그래프 그려보기
par(mfrow=c(2,2))
plot(f_m)

## 잔차들이 골고루 퍼져있으며
## QQ 검증도를 보면 정규성을 띄는 것

#####################

# 2. 회귀 모델 검증 
# fitted function 이용해 전체의 추정값 도출 
y_hat <- fitted(f_m)

# 실제값 
y <- new_data[['MonthlyIncome']]

# 추정값 vs 실제값 그래프 
par(mfrow=c(1,2))
scatter.smooth(y_hat,y, main='추정값 vs 실제값', xlab='추정값', ylab='실제값')


## 그중(설명변수) 하나 뽑아서 상관관계 그래프 그려봄
scatter.smooth(new_data$JobLevel,new_data$MonthlyIncome, xlab = '숙련도', ylab = '수입', main='숙련도와 수입의 상관도')


## 그래프가 이렇게 나오는 이유는 설명변수들중에 명목형 변수들이 많이 있기에 띄엄띄엄 나오는 것.


# 추가적으로 인공신경망과 선형회귀 비교

# 인공신경망 모델 만들기 hidden node = 5 개 
ai <- nnet(MonthlyIncome~., data = new_data, size = 5, decay=0.1, linout=T)

# 인공신경망 모델로 예측값 생성
ai_pred <- predict(ai, newdata = new_data)

# 인공 신경망과 실제값 비교
scatter.smooth(ai_pred, y)

# 모델 비교를 위한 데이터 프레임
# 인공신경망 추정값과 회귀모델 추정값 실제값 비교 

# 실제값, 인공신경망 모델, 회귀 모델로 데이터 프레임만들고 
model_cpr <- data.frame('실제값' = y, '인공신경망 모델'=ai_pred[,1], '회귀모델' = y_hat)
head(model_cpr)

# 실제값과 각 모델의 잔차를 절대값으로 계산하고
cpr <- data.frame('인공신경망 잔차' = abs(model_cpr[,1]- model_cpr[,2]), '회귀모델 잔차' = abs(model_cpr[,1] - model_cpr[,3]))

# 인공신경망 추정값의 잔차가 더 클 경우 0, 작을경우 1 부여
cpr2 <- ifelse(cpr[,1] >=cpr[,2], 0 , 1)

cpr <- cbind(cpr,cpr2)

# 총 개수 리턴
cpr <- cbind(cpr,cpr2)
cpr_count <- cpr %>% group_by(cpr2) %>% summarise(count = n())
cpr_count

## 아직은 인공신경망이 더 부정확하다는 것을 알 수 있음..

## hidden 노드를 2층으로 c(3,3)으로 수정해서 보다 정확한 추정치 도출
## 하려고하였으나 사이즈가 커지면 R이 꺼져서 코드로만 남김.
# ai_2 <- nnet(MonthlyIncome~., data = new_data, size = c(3,3), decay=0.1, linout=T)
# ai_pred_2 <- predict(ai_2, newdata = new_data)
# 
# class(new_data)
# 
# predict(m, newdata=iris)




