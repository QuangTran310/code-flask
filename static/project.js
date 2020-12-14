function myFunction() {
    var x = document.getElementsByClassName('my_tag');
    console.log(x)
    if (x.style.visibility === 'hidden') {
        x.style.visibility = 'visible';
    } else {
        x.style.visibility = 'hidden';
    }
}