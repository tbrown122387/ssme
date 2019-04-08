library(quantmod)
getSymbols("SPY")
rets <- 100*diff(log(Ad(SPY)))[-1]
write.table(rets, "ssme/example/spy_returns.csv", quote = F, row.names = F, col.names = F)
