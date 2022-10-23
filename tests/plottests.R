library(stringr)

df <- read.csv('./out/results.csv')
df$gap <- as.numeric(format(round(as.numeric(df$gap), 2), nsmall=2))
df_min <- aggregate(.~file, df, min, na.action=na.pass)
df_avg <- aggregate(.~file, df, mean, na.action=na.pass)

df_r <- data.frame(df_min, df_avg[2:3])
colnames(df_r) <- c('instance', 'ms_*', 'gap_*', 'ms_avg', 'gap_avg')
write.csv(df_r, './out/means.csv', row.names=FALSE, quote=FALSE)

options(scipen=10)
p_sw <- df[1, ]['file']
df_px <- data.frame()

for (i in 1:nrow(df)) {
    sw <- df[i, ]['file']
    df_n <- df[i, c('file', 'makespan')]

    if (grepl(substr(p_sw, 1, 1), sw)) {
        df_px <- rbind(df_px, df_n)
        
    } else {
        name <- sub("^([[:alpha:]]*).*", "\\1", df_px[1, ]['file'])
        png(str_interp('./docs/${name}.png'))
        
        boxplot(
            makespan ~ file, 
            data=df_px, 
            main=str_interp('Makespan para as instâncias ${name}'), 
            xlab='', 
            ylab=''
        )

        stripchart(
            df_px$makespan ~ df_px$file, 
            vertical=TRUE, 
            method='jitter',
            pch=19, 
            add=TRUE, 
            col=1:length(levels(as.factor(df$file)))
        )

        dev.off()
        df_px <- df_n
    }

    p_sw <- sw
}