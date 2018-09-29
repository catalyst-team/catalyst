## Codestyle

tl;dr:
- right margin - 80
- double quotes
- full names: 
    - `model`, `criterion`, `optimizer`, `scheduler` - okay 
    - `mdl`,`crt`, `opt`, `scd` - not okay
- long names solution
    - okay:
    ```bash
    def my_pure_long_name(
            self,
            model, criterion=None, optimizer=None, scheduler=None,
            debug=True):
        """code"""
    ```
    - not okay:
    ```bash
    def my_pure_long_name(self,
                          model, criterion=None, optimizer=None, scheduler=None,
                          debug=True):
        """code"""
    ```
    - why? name refactoring. with first one solution, 
            there are no problems with pep8 codestyle check.
- \* in funcs for force key-value args
