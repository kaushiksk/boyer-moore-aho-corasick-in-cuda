#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <iostream>

using namespace std;
 
# define NO_OF_CHARS 256
 
// A utility function to get maximum of two integers
int max (int a, int b) { return (a > b)? a: b; }
 
// The preprocessing function for Boyer Moore's
// bad character heuristic
void badCharHeuristic( char *str, int size, 
                        int badchar[NO_OF_CHARS])
{
    int i;
 
    // Initialize all occurrences as -1
    for (i = 0; i < NO_OF_CHARS; i++)
         badchar[i] = -1;
 
    // Fill the actual value of last occurrence 
    // of a character
    for (i = 0; i < size; i++)
         badchar[(int) str[i]] = i;
}
 

// preprocessing for strong good suffix rule
void preprocess_strong_suffix(int *shift, int *bpos,
                                char *pat, int m)
{
    // m is the length of pattern 
    int i=m, j=m+1;
    bpos[i]=j;
 
    while(i>0)
    {
        /*if character at position i-1 is not equivalent to
          character at j-1, then continue searching to right
          of the pattern for border */
        while(j<=m && pat[i-1] != pat[j-1])
        {
            /* the character preceding the occurence of t in 
               pattern P is different than mismatching character in P, 
               we stop skipping the occurences and shift the pattern
               from i to j */
            if (shift[j]==0)
                shift[j] = j-i;
 
            //Update the position of next border 
            j = bpos[j];
        }
        /* p[i-1] matched with p[j-1], border is found.
           store the  beginning position of border */
        i--;j--;
        bpos[i] = j; 
    }
}
 
//Preprocessing for case 2
void preprocess_case2(int *shift, int *bpos,
                      char *pat, int m)
{
    int i, j;
    j = bpos[0];
    for(i=0; i<=m; i++)
    {
        /* set the border postion of first character of pattern
           to all indices in array shift having shift[i] = 0 */
        if(shift[i]==0)
            shift[i] = j;
 
        /* suffix become shorter than bpos[0], use the position of 
           next widest border as value of j */
        if (i==j)
            j = bpos[j];
    }
}

void search( char *text,  char *pat)
{

 
    int badchar[NO_OF_CHARS];
    int  j;
    int m = strlen(pat);
    int n = strlen(text);
 
    int bpos[m+1], shift[m+1];
 
    //initialize all occurence of shift to 0
    for(int i=0;i<m+1;i++) shift[i]=0;
 
    //do preprocessing
    preprocess_strong_suffix(shift, bpos, pat, m);
    preprocess_case2(shift, bpos, pat, m);
 
    /* Fill the bad character array by calling 
       the preprocessing function badCharHeuristic() 
       for given pattern */
    badCharHeuristic(pat, m, badchar);


    for(int i=0;i<m+1;i++)
      printf("%d ",shift[i]);
    cout<<endl;
    for(int i=0;i<NO_OF_CHARS;i++)
      printf("%d ",badchar[i]);
    cout<<endl;

 
    int s = 0;  // s is shift of the pattern with 
                // respect to text
    while(s <= (n - m))
    {
        int j = m-1;
 
        /* Keep reducing index j of pattern while 
           characters of pattern and text are 
           matching at this shift s */
        while(j >= 0 && pat[j] == text[s+j])
            j--;
 
        /* If the pattern is present at current
           shift, then index j will become -1 after
           the above loop */
        printf("%d %d %d %d\n",j+1,s+j,shift[j+1],j - badchar[text[s+j]]);
        if (j < 0)
        {
            printf("\n pattern occurs at shift = %d\n", s);
 
            /* Shift the pattern so that the next 
               character in text aligns with the last 
               occurrence of it in pattern.
               The condition s+m < n is necessary for 
               the case when pattern occurs at the end 
               of text */
            s += shift[0];
 
        }
        else
            s += max(shift[j+1] , j - badchar[text[s+j]]);
          //printf("%d \n",s);
    }
}

//Driver 
int main()
{
    char text[100];
    char pat[100];

    cin>>text>>pat;

    search(text, pat);
    return 0;
}




