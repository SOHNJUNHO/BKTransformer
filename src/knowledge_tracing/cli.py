from .main import build_model, load_dataframe, main, parse_args

__all__ = ["build_model", "load_dataframe", "main", "parse_args"]


if __name__ == "__main__":
    import pytorch_lightning as pl

    cli_args = parse_args()
    pl.seed_everything(cli_args.seed, workers=True)
    main(cli_args)
