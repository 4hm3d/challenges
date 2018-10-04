from datetime import datetime, timedelta

born_date = "11/29/1991"
current_date = datetime.now()
parts = born_date.split("/")

print(
    str(
        (
            ( ( current_date.year - int(parts[2]) ) * 365.2422 ) + # days in a year
            ( 365.2422/12*int(parts[0]) ) +
            ( int(parts[1]) )
        ) / 365.2422
    ) + " Years old!"
)